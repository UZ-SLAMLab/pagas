#!/usr/bin/env python3
# License: CC BY-NC 4.0 - see /LICENSE
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import open3d as o3d
from PIL import Image
from pycolmap import Reconstruction
from scipy.ndimage import binary_erosion


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from datareader.utils import quaternion_to_rotation_matrix
except Exception:
    quaternion_to_rotation_matrix = None

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_mask_png(mask_bool, path_png):
    Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(path_png)

def make_relative_save_path(image_name, new_ext):
    rel = image_name.replace("\\", "/")
    base, _ = os.path.splitext(rel)
    return base + new_ext

def load_colmap(colmap_sparse_folder):
    colmap_dir = os.path.join(colmap_sparse_folder, "0")
    if not os.path.exists(colmap_dir):
        colmap_dir = colmap_sparse_folder
    assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist."

    rec = Reconstruction(colmap_dir)
    if len(rec.images) == 0:
        raise ValueError("No images found in COLMAP reconstruction.")

    w2c_list, image_ids = [], []
    K_dict, size_dict = {}, {}

    single_cam_for_all = len(rec.images) > 1 and len(rec.cameras) == 1
    shared_cam = next(iter(rec.cameras.values())) if single_cam_for_all else None

    for image_id, img in rec.images.items():
        cam = shared_cam if single_cam_for_all else rec.cameras[img.camera_id]
        pose_cw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world

        # Robust way to get rotation, avoid guessing quaternion order
        if hasattr(pose_cw.rotation, "matrix"):
            R = pose_cw.rotation.matrix()
        else:
            # Fallback only if needed
            assert quaternion_to_rotation_matrix is not None, "Need quaternion_to_rotation_matrix fallback"
            q = pose_cw.rotation.quat  # check your helper expects this order
            R = quaternion_to_rotation_matrix(q)

        t = pose_cw.translation.reshape(3, 1)
        w2c = np.vstack([np.hstack([R, t]), np.array([[0.0, 0.0, 0.0, 1.0]])])
        w2c_list.append(w2c)
        image_ids.append(image_id)

        fx, fy = cam.focal_length_x, cam.focal_length_y
        # Use COLMAP convention. Principal point is in pixels from top left corner
        # Use centers at u+0.5 and v+0.5 when creating rays, so no 0.5 shift here
        cx, cy = cam.principal_point_x, cam.principal_point_y
        K_dict[image_id] = np.array([[fx, 0.0, cx],
                                     [0.0, fy, cy],
                                     [0.0, 0.0, 1.0]], dtype=np.float64)
        size_dict[image_id] = (cam.width, cam.height)

    w2c_all = np.stack(w2c_list, axis=0)
    c2w_all = np.linalg.inv(w2c_all)

    image_names = [rec.images[iid].name for iid in image_ids]
    order = np.argsort(np.array(image_names, dtype=object))
    image_names_sorted = [image_names[i] for i in order]
    camtoworld_sorted = c2w_all[order]
    image_ids_sorted = [image_ids[i] for i in order]

    if single_cam_for_all:
        cam_model_name = shared_cam.model.name
    else:
        last_img_id = image_ids_sorted[-1]
        cam_model_name = rec.cameras[rec.images[last_img_id].camera_id].model.name
    if cam_model_name not in [0, "SIMPLE_PINHOLE", 1, "PINHOLE", 2]:
        print("Warning, camera model has distortion, distortion is ignored.")

    return image_names_sorted, camtoworld_sorted, K_dict, size_dict, image_ids_sorted, cam_model_name

def build_raycast_scene(mesh_file):
    mesh_legacy = o3d.io.read_triangle_mesh(mesh_file)
    if mesh_legacy is None or len(mesh_legacy.triangles) == 0:
        raise ValueError(f"Failed to read a valid mesh from {mesh_file}")
    if not mesh_legacy.has_vertex_normals():
        mesh_legacy.compute_vertex_normals()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)
    return scene

def make_rays_opencv(K_3x3, c2w_4x4, width, height):
    """
    Build per pixel rays in world coordinates using OpenCV camera model.
    Returns rays tensor H W 6 and the per pixel z component of the unit camera ray d_cv_z.
    Rays are origins in world, directions in world, both float32.
    """
    fx, fy = K_3x3[0, 0], K_3x3[1, 1]
    cx, cy = K_3x3[0, 2], K_3x3[1, 2]

    # Pixel centers
    u = np.arange(width, dtype=np.float32) + 0.5
    v = np.arange(height, dtype=np.float32) + 0.5
    uu, vv = np.meshgrid(u, v)  # H W

    x = (uu - cx) / fx
    y = (vv - cy) / fy
    d_cv = np.stack([x, y, np.ones_like(x)], axis=-1)  # H W 3
    d_cv = d_cv / np.linalg.norm(d_cv, axis=-1, keepdims=True)  # unit length
    d_cv_z = d_cv[..., 2].copy()  # save z component for z depth conversion

    R = c2w_4x4[:3, :3]
    C = c2w_4x4[:3, 3]
    d_world = d_cv @ R.T  # H W 3, still unit length
    origins = np.broadcast_to(C.reshape(1, 1, 3), d_world.shape)

    rays = np.concatenate([origins, d_world], axis=-1).astype(np.float32)  # H W 6
    return rays, d_cv_z

def render_depth_for_camera(scene,
                            K_cv_3x3,
                            c2w_cv_4x4,
                            width,
                            height,
                            erode_mask_iters=1):
    """
    Returns:
      depth_m: H W float32 in meters, z in the OpenCV camera
      mask:    H W bool
    """
    rays_np, dcv_z = make_rays_opencv(K_cv_3x3, c2w_cv_4x4, width, height)
    rays = o3d.core.Tensor(rays_np)  # H W 6 float32

    ans = scene.cast_rays(rays)
    t_hit = ans['t_hit'].numpy().reshape(height, width)  # path length in meters along unit ray
    hit_mask = np.isfinite(t_hit)

    # Convert path length to z depth in the camera
    depth_m = t_hit * dcv_z
    depth_m = np.where(hit_mask & np.isfinite(depth_m) & (depth_m > 0.0), depth_m, 0.0)

    if erode_mask_iters > 0:
        structure = np.ones((3, 3), dtype=bool)
        clean_mask = binary_erosion(hit_mask, structure=structure, iterations=erode_mask_iters)
        depth_m = np.where(clean_mask, depth_m, 0.0)
        hit_mask = clean_mask

    return depth_m.astype(np.float32), hit_mask


def main():
    parser = argparse.ArgumentParser(description="Render z depth from a mesh at COLMAP poses")
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to triangle mesh file readable by Open3D')
    parser.add_argument('--colmap_sparse_folder', type=str, required=True, help='Path to COLMAP /sparse folder.')
    parser.add_argument('--output_depth_dir', type=str, required=True, help='Output depth folder')
    parser.add_argument('--output_masks_dir', type=str, required=True, help='Output mask folder')
    parser.add_argument('--erode_iters', type=int, default=1, help='Binary erosion iterations for the mask')
    args = parser.parse_args()

    depth_dir = args.output_depth_dir
    mask_dir = args.output_masks_dir
    ensure_dir(depth_dir)
    ensure_dir(mask_dir)

    image_names, camtoworld_all, K_dict, size_dict, image_ids, cam_model = load_colmap(args.colmap_sparse_folder)
    print(f"Loaded {len(image_names)} images from COLMAP. Camera model: {cam_model}")

    scene = build_raycast_scene(args.mesh_file)

    for idx in tqdm(range(len(image_names)), desc="Rendering"):
        image_id = image_ids[idx]
        name_rel = image_names[idx]
        width, height = size_dict[image_id]
        K_cv = K_dict[image_id]
        c2w_cv = camtoworld_all[idx]

        depth_m, mask = render_depth_for_camera(scene, K_cv, c2w_cv, width, height, erode_mask_iters=max(0, int(args.erode_iters)))

        rel_depth_npz = make_relative_save_path(name_rel, ".npz")
        rel_mask_png = make_relative_save_path(name_rel, ".png")

        out_depth_npz = os.path.join(depth_dir, rel_depth_npz)
        out_mask_png = os.path.join(mask_dir, rel_mask_png)

        ensure_dir(os.path.dirname(out_depth_npz))
        ensure_dir(os.path.dirname(out_mask_png))

        np.savez(out_depth_npz, depth_m)
        save_mask_png(mask, out_mask_png)

if __name__ == "__main__":
    main()
