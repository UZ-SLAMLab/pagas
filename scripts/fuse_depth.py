# License: CC BY-NC 4.0 - see /LICENSE
import os, sys, gc, heapq, argparse, ctypes
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_erosion
import open3d as o3d
from pycolmap import Reconstruction

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datareader.colmap import get_mask_from_path
from datareader.utils import quaternion_to_rotation_matrix


libc = ctypes.CDLL(None)

# ----------------------------- Args -----------------------------
ap = argparse.ArgumentParser("Fuse predicted depths with Scalable TSDF (robust)")
ap.add_argument('--data_dir', type=str, required=True,
                help='Path to RESULTS for this scan (e.g. .../results/depth_init_pagas)')
ap.add_argument('--relative_path_to_colmap', type=str, default='',
                help="Optional subdir under scan root that holds COLMAP sparse (default tries common locations).")
ap.add_argument('--image_name', type=str, default='images',
                help='Subfolder name for RGB images under scan root.')
ap.add_argument('--depth_name', type=str, default='depth',
                help='Subfolder name under data_dir with predicted depths (.npz/.npy).')
ap.add_argument('--mask_name', type=str, default='',
                help='Subfolder name under data_dir with masks (optional).')
ap.add_argument('--depth_trunc', type=float, default=-1.0,
                help='Depth max in scene units (<=0 => derived from camera bbox diameter).')
ap.add_argument('--voxel_size', type=float, default=-1.0,
                help='Voxel size in scene units (<=0 => depth_trunc/mesh_res).')
ap.add_argument('--sdf_trunc', type=float, default=5.0,
                help='TSDF truncation = sdf_trunc * voxel_size.')
ap.add_argument('--mesh_res', type=int, default=1024,
                help='Used only when voxel_size <= 0 (voxel_size = depth_trunc/mesh_res).')
ap.add_argument('--min_mesh_size', type=int, default=50,
                help='Postprocess: keep connected components with >= faces.')
ap.add_argument('--num_cluster', type=int, default=1,
                help='Postprocess: keep the N largest components (ties included).')
ap.add_argument('--mesh_name', type=str, default='mesh.ply',
                help='Base mesh filename; suffixes are added with parameters.')
ap.add_argument('--erode_borders', action='store_true',
                help='If set, erode depth borders before fusion.')
args = ap.parse_args()


# ------------------------ Utilities -----------------------------
def post_process_mesh(base: o3d.geometry.TriangleMesh, *, min_faces=50, parts_to_keep=1):
    vertices = np.asarray(base.vertices)
    faces = np.asarray(base.triangles)
    has_color = base.has_vertex_colors()
    has_normal = base.has_vertex_normals()
    if has_color: colors = np.asarray(base.vertex_colors)
    if has_normal: normals = np.asarray(base.vertex_normals)

    triangle_labels, _, _ = base.cluster_connected_triangles()
    labels = np.asarray(triangle_labels, dtype=np.int32)
    counts = np.bincount(labels)
    large = np.flatnonzero(counts >= min_faces)
    if large.size > parts_to_keep:
        kth = heapq.nlargest(parts_to_keep, counts[large])[-1]
        large = large[counts[large] >= kth]
    keep_mask = np.isin(labels, large, assume_unique=True)
    del labels; gc.collect()
    if not keep_mask.any():
        return o3d.geometry.TriangleMesh()

    faces_kept = faces[keep_mask]
    unique_v, inverse = np.unique(faces_kept, return_inverse=True)
    new_vertices = vertices[unique_v]
    new_faces = inverse.reshape(faces_kept.shape)
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(new_vertices.copy())
    out.triangles = o3d.utility.Vector3iVector(new_faces.astype(np.int32))
    if has_color: out.vertex_colors = o3d.utility.Vector3dVector(colors[unique_v].copy())
    if has_normal: out.vertex_normals = o3d.utility.Vector3dVector(normals[unique_v].copy())
    out.remove_unreferenced_vertices()
    out.remove_degenerate_triangles()
    return out


def compute_bbox_center_radius(camtoworld):
    cam_pos = camtoworld[:, :3, 3]
    bbox_min, bbox_max = cam_pos.min(0), cam_pos.max(0)
    center = (bbox_min + bbox_max) / 2.0
    radius = np.linalg.norm(cam_pos - center, axis=1).max()
    return center, float(radius)


def _stem(p: str) -> str:
    b = os.path.basename(p)
    return os.path.splitext(b)[0]


def format5(x: float) -> str:
    # Exact zero as "0"
    if x == 0.0:
        return "0"

    # Fixed decimal with enough precision
    s = f"{x:.10f}"

    # Separate sign
    sign = ""
    if s.startswith("-"):
        sign = "-"
        s = s[1:]

    if "." in s:
        int_part, frac = s.split(".", 1)
    else:
        int_part, frac = s, ""

    # First 5 decimal digits, padded
    frac5 = (frac + "00000")[:5]

    # If integer part is nonzero, always decimal
    # If integer part is zero but there is some nonzero digit in the first 5 decimals, also decimal
    if int_part != "0" or any(c != "0" for c in frac5):
        frac5_trim = frac5.rstrip("0")
        if frac5_trim:
            return f"{sign}{int_part}.{frac5_trim}"
        else:
            return f"{sign}{int_part}"
    else:
        # Scientific notation for very small numbers
        sci = f"{x:.10e}"

        sign2 = ""
        if sci.startswith("-"):
            sign2 = "-"
            sci = sci[1:]

        mant, exp = sci.split("e", 1)
        exp_str = "e" + exp

        if "." in mant:
            mant_int, mant_frac = mant.split(".", 1)
        else:
            mant_int, mant_frac = mant, ""

        digits = mant_int + mant_frac
        digits5 = (digits + "00000")[:5]
        digits5 = digits5.rstrip("0") or "0"

        if len(digits5) > 1:
            return f"{sign2}{digits5[0]}.{digits5[1:]}{exp_str}"
        else:
            return f"{sign2}{digits5}{exp_str}"


# ---------------------- Resolve directories ----------------------
results_dir = Path(args.data_dir).resolve()                               
assert results_dir.is_dir(), f"   data_dir not found: {results_dir}"
scan_root = results_dir                                     

image_dir = (scan_root / args.image_name).resolve()                       
depth_dir = (results_dir / args.depth_name).resolve()                     
mask_dir  = (results_dir / args.mask_name).resolve() if args.mask_name else None

# COLMAP sparse (prefer undistorted locations)
if args.relative_path_to_colmap:
    parent_sparse = (scan_root / args.relative_path_to_colmap).resolve()
else:
    parent_sparse = scan_root
colmap_dir = None
for cand in ["sparse/0", "sparse"]:
    p = (parent_sparse / cand).resolve()
    if p.is_dir():
        colmap_dir = p; break

missing = []
for lbl, p in [("images", image_dir), ("depth", depth_dir)]:
    if not p.is_dir(): missing.append(f"{lbl}: {p}")
if mask_dir and not mask_dir.is_dir(): missing.append(f"mask: {mask_dir}")
if not colmap_dir or not colmap_dir.is_dir(): missing.append(f"COLMAP sparse not found under {parent_sparse}/sparse[/0]")
if missing:
    raise FileNotFoundError("   Required folders not found:\n  " + "\n  ".join(map(str, missing)))

# ----- Load COLMAP intrinsics & poses  -----
rec = Reconstruction(str(colmap_dir))
if len(rec.images) == 0:
    raise ValueError("  No images in COLMAP reconstruction.")

names = []
Twc_list = []
K_by_name = {}
size_by_name = {}

def quat_xyzw_to_R(q_xyzw: np.ndarray) -> np.ndarray:
    # q = [x,y,z,w]
    x, y, z, w = map(float, q_xyzw)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=np.float64)

def rigid3d_to_4x4(Tcw_obj):
    """Return 4x4 Tcw from a pycolmap Rigid3d or compatible object."""
    # Preferred path
    M = getattr(Tcw_obj, "matrix", None)
    if callable(M):
        M = np.asarray(M(), dtype=np.float64)
        if M.shape == (3, 4):
            M = np.vstack([M, np.array([0, 0, 0, 1], dtype=np.float64)])
        elif M.shape != (4, 4):
            raise ValueError(f"   Unexpected Tcw matrix shape: {M.shape}")
        return M

    # Fallback: build from rotation + translation
    rot = getattr(Tcw_obj, "rotation", None)
    trans = getattr(Tcw_obj, "translation", None)
    if rot is None or trans is None:
        raise TypeError("   Rigid3d has neither .matrix() nor (rotation, translation)")

    # Get quaternion; different pycolmap versions expose .quat or .coeffs
    q = getattr(rot, "quat", None)
    if q is None:
        q = getattr(rot, "coeffs", None)
    if q is None:
        q = rot  # last resort

    q = np.asarray(q, dtype=np.float64).reshape(4,)
    # pycolmap usually stores as [qw, qx, qy, qz]; we expect [x, y, z, w]
    qw, qx, qy, qz = q
    q_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)

    R = quaternion_to_rotation_matrix(q_xyzw)
    t = np.asarray(trans, dtype=np.float64).reshape(3, 1)
    Tcw = np.eye(4, dtype=np.float64)
    Tcw[:3, :3] = R
    Tcw[:3, 3] = t[:, 0]
    return Tcw

for img_id, img in rec.images.items():
    cam = rec.cameras[img.camera_id]

    pose_cw = getattr(img, "cam_from_world", None)
    if pose_cw is not None:
        Tcw_obj = pose_cw() if callable(pose_cw) else pose_cw
        Tcw = rigid3d_to_4x4(Tcw_obj)           
    else:
        # Very old pycolmap fallback: qvec/tvec on the image itself
        qvec = getattr(img, "qvec", None)  # [qw, qx, qy, qz]
        tvec = getattr(img, "tvec", None)
        assert qvec is not None and tvec is not None, "No pose available on image."
        qw, qx, qy, qz = map(float, qvec)
        q_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
        R = quaternion_to_rotation_matrix(q_xyzw)
        Tcw = np.eye(4, dtype=np.float64)
        Tcw[:3, :3] = R
        Tcw[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)

    Twc = np.linalg.inv(Tcw).astype(np.float64)

    # --- intrinsics K ---
    fx = getattr(cam, "focal_length_x", None)
    fy = getattr(cam, "focal_length_y", None)
    cx = getattr(cam, "principal_point_x", None)
    cy = getattr(cam, "principal_point_y", None)
    if None in (fx, fy, cx, cy):
        # Older pycolmap: use cam.model + cam.params
        model = cam.model.name if hasattr(cam.model, "name") else cam.model
        pars = np.asarray(cam.params, dtype=np.float64)
        if model == "SIMPLE_PINHOLE" or model == 0:
            fx = fy = pars[0]; cx = pars[1]; cy = pars[2]
        elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", 1, 2):
            fx, fy, cx, cy = pars[:4]
        else:
            # Fallback: assume first four are fx,fy,cx,cy
            fx, fy, cx, cy = pars[:4]
    # Open3D center-of-pixel convention
    cx, cy = float(cx) - 0.5, float(cy) - 0.5
    K = np.array([[float(fx), 0.0, cx],
                  [0.0, float(fy), cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    w = int(cam.width); h = int(cam.height)

    name = img.name
    names.append(name)
    Twc_list.append(Twc)
    K_by_name[name] = K
    size_by_name[name] = (w, h)

# sort and stack
order = np.argsort(np.asarray(names, dtype=object))
names = [names[i] for i in order]
Twc_list = [Twc_list[i] for i in order]
camtoworld = np.stack(Twc_list, axis=0).astype(np.float64)

# ------------------ Pair images with depths ---------------------
img_files = {Path(p).name: str(image_dir / p) for p in os.listdir(image_dir) if (image_dir / p).is_file()}
depth_candidates = [str(p) for p in depth_dir.glob("*.npz")] + [str(p) for p in depth_dir.glob("*.npy")]
depth_by_name  = {Path(p).name: p for p in depth_candidates}
depth_by_stem  = {_stem(p): p for p in depth_candidates}

pairs = []
for name in names:
    if name in img_files:
        ip = img_files[name]
    else:
        raise FileNotFoundError(f"   Image file missing on disk for COLMAP name: {name}")
    dp = depth_by_name.get(name, None)
    if dp is None:
        st = Path(name).stem
        dp = depth_by_stem.get(st, None)
    if dp is None:
        raise FileNotFoundError(f"   No depth file matching image {name} (by exact name or stem).")
    pairs.append((ip, dp, name))
print(f"   [info] matched {len(pairs)} frames")

# ------------------ Derive depth_trunc/voxels -------------------
if args.depth_trunc <= 0:
    _, radius = compute_bbox_center_radius(camtoworld)
    depth_trunc = 2.0 * radius
else:
    depth_trunc = float(args.depth_trunc)

if args.voxel_size <= 0:
    voxel_size = float(depth_trunc / max(1, int(args.mesh_res)))
else:
    voxel_size = float(args.voxel_size)

sdf_trunc = float(args.sdf_trunc) * voxel_size

# output filenames
depth_str = format5(depth_trunc)
vox_str   = format5(voxel_size)
sdf_str   = format5(args.sdf_trunc)

mesh_name = args.mesh_name.replace(
    ".ply",
    f"_trunc{depth_str}_vox{vox_str}_sdf{sdf_str}.ply"
)

# ------------------ Build TSDF (Scalable, robust) ----------------
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size,
    sdf_trunc=sdf_trunc,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

# ------------------------ Fuse loop ------------------------------
for i, (img_path, depth_path, name) in tqdm(enumerate(pairs), total=len(pairs), desc="   TSDF integration"):
    # load depth
    if depth_path.endswith(".npz"):
        npz = np.load(depth_path)
        depth = npz["depth"] if "depth" in npz.files else npz[npz.files[0]]
    else:
        depth = np.load(depth_path)

    # dtype/shape for depth
    depth = np.asarray(depth, dtype=np.float32, order="C").squeeze()
    assert depth.ndim == 2, f"Depth must be HxW, got {depth.shape}"
    np.nan_to_num(depth, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.clip(depth, 0.0, float(depth_trunc), out=depth)

    # load color as RGB uint8
    color = np.asarray(Image.open(img_path).convert("RGB"))
    color = np.asarray(color, dtype=np.uint8, order="C")

    # optional mask
    if args.mask_name:
        mp = str(mask_dir / name)
        if os.path.isfile(mp):
            mask = get_mask_from_path(filepath=mp)[:, :, 0].astype(bool)
            if mask.shape != depth.shape:
                mask = np.array(Image.fromarray((mask.astype(np.uint8)*255)).resize(depth.shape[::-1], Image.NEAREST)) > 0
        else:
            mask = None
    else:
        mask = None

    # sync color to depth size if needed
    if color.shape[:2] != depth.shape:
        color = np.array(Image.fromarray(color).resize(depth.shape[::-1], Image.BILINEAR))
        color = np.asarray(color, dtype=np.uint8, order="C")

    # scale intrinsics to depth resolution
    w0, h0 = size_by_name[name]
    fx, fy, cx, cy = K_by_name[name][0,0], K_by_name[name][1,1], K_by_name[name][0,2], K_by_name[name][1,2]
    H, W = depth.shape
    if (W != w0) or (H != h0):
        sx, sy = (W / float(w0)), (H / float(h0))
        fx *= sx; fy *= sy
        cx = (cx + 0.5) * sx - 0.5
        cy = (cy + 0.5) * sy - 0.5
        w0, h0 = W, H

    # erode borders + mask
    if args.erode_borders:
        thickness = 10
        if thickness > 0:
            depth[:thickness, :] = 0.0
            depth[-thickness:, :] = 0.0
            depth[:, :thickness] = 0.0
            depth[:, -thickness:] = 0.0
            if mask is not None:
                m = binary_erosion(mask, structure=np.ones((5,5), bool))
                depth[~m] = 0.0
    else:
        if mask is not None:
            depth[~mask] = 0.0 

    # final Open3D objects
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(W), height=int(H),
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy)
    )
    extrinsic = np.asarray(np.linalg.inv(camtoworld[i]), dtype=np.float64, order="C")
    assert extrinsic.shape == (4,4) and np.isfinite(extrinsic).all()

    depth_o3d = o3d.geometry.Image(depth)   
    color_o3d = o3d.geometry.Image(color)   
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,
        depth_trunc=float(depth_trunc),
        convert_rgb_to_intensity=False
    )

    try:
        volume.integrate(rgbd, intrinsic, extrinsic)
    except Exception as e:
        raise RuntimeError(f"   Open3D integrate failed at frame {i} ({name})") from e

# ----------------------- Extract & save --------------------------
mesh = volume.extract_triangle_mesh()
out_mesh = results_dir / mesh_name
o3d.io.write_triangle_mesh(str(out_mesh), mesh)
print(f"   Mesh saved at {out_mesh}")

mesh_post = post_process_mesh(mesh, min_faces=int(args.min_mesh_size), parts_to_keep=int(args.num_cluster))
out_post = results_dir / mesh_name.replace('.ply', '_post.ply')
o3d.io.write_triangle_mesh(str(out_post), mesh_post)
print(f"   Mesh post processed saved at {out_post}")
