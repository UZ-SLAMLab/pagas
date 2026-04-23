# License: CC BY-NC 4.0 - see /LICENSE
import os
import torch
import struct
import numpy as np
from pathlib import Path
from typing import List, Dict
from PIL import Image
import shutil

from pagas.geo_ops import depth_to_world_normal


def data_to_cuda(data, scale_factor, cfg, device):
    ds = data[scale_factor]

    # Required tensors
    K = ds["K"].to(device)                    # [B, 3, 3]
    context_K = ds["context_K"].to(device)    # [B, M, 3, 3]
    all_pixels = ds["all_image"].squeeze(0).to(device).float() / 255.0  # [M+1, H, W, 3], assumes B = 1
    depth_init = ds["depth_init"].to(device)  # [B, H, W]

    # Optional tensors
    all_mask = None
    if "all_mask" in ds:
        all_mask = ds["all_mask"].to(device).squeeze()

    gt_depth = None
    if "gt_depth" in ds:
        gt_depth = ds["gt_depth"].to(device).squeeze(0)

    inv_color_grad_weight = None
    if "inv_color_grad_weight" in ds:
        inv_color_grad_weight = ds["inv_color_grad_weight"].to(device).squeeze(0)

    # Rotation
    rotation90 = torch.tensor(
        [
            [0., -1., 0., 0.],
            [1.,  0., 0., 0.],
            [0.,  0., 1., 0.],
            [0.,  0., 0., 1.],
        ],
        device=device,
    )
    T = rotation90 if data.get("rotated_image", False) else torch.eye(4, device=device)

    # Ground truth normals
    gt_normal = None
    if gt_depth is not None:
        gt_normal = depth_to_world_normal(
            gt_depth.unsqueeze(-1),
            T,
            K.squeeze(0),
            method=cfg.depth_to_normal_method,
        ).squeeze(0)

    return (
        K,
        context_K,
        all_pixels,
        all_mask,
        gt_depth,
        gt_normal,
        inv_color_grad_weight,
        depth_init,
    )


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion from XYZW format to a 3x3 rotation matrix.
    Parameters:
    q (numpy.array): Quaternion in XYZW format, shape (4,)
    Returns:
    R (numpy.array): Rotation matrix, shape (3, 3)
    """
    # Check if the quaternion is normalized and normalize it if necessary
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0):
        q /= norm
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R


def get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def get_mask_from_path(filepath: Path) -> np.ndarray:
    """
    Utility function to read a mask image from the given path and return a boolean numpy array
    """
    pil_mask = Image.open(filepath)
    mask = np.array(pil_mask)[..., None].astype(bool)
    if len(mask.shape) == 4:
        mask = mask[:, :, 0, :]
    return mask


def get_indices(file_path: Path, num_context_views: int = -1) -> Dict:
    """
    Parse the views.cvg file and return a dictionary where each key is an image index
    starting at 0 and increasing by 1, and its value is a list of adjusted closest image indices.
    """
    raw_targets = []
    raw_contexts = []
    if num_context_views == -1:
        num_context_views = int(1e5)

    with open(file_path, 'r') as file:
        for target_line, context_line in zip(file, file):
            target_idx = int(os.path.splitext(target_line.strip())[0])
            context_list = [
                int(os.path.splitext(name)[0])
                for name in context_line.strip().split(', ')[:num_context_views]
            ]
            raw_targets.append(target_idx)
            raw_contexts.append(context_list)

    # Build a mapping from original indices to new indices starting at 0
    unique_indices = sorted(set(raw_targets))
    index_map = {orig: new for new, orig in enumerate(unique_indices)}

    # Create final dictionary with new indices
    final_indices = {
        index_map[target]: [index_map[c] for c in context if c in index_map]
        for target, context in zip(raw_targets, raw_contexts)
    }

    return dict(sorted(final_indices.items()))


def ensure_colmap_sparse_model(input_dir: str | Path) -> None:
    """
    Ensure a COLMAP sparse model in `input_dir` has a valid points3D file.

    Rules:
      - At least one of the pairs (cameras.bin, images.bin) or (cameras.txt, images.txt)
        must exist, otherwise raise FileNotFoundError.
      - If the binary pair exists and points3D.bin is missing, create a valid
        empty points3D.bin (uint64 0 as number of points).
      - If the text pair exists and points3D.txt is missing, create an empty
        or header-only points3D.txt.
    """
    input_dir = Path(input_dir)

    cameras_bin = input_dir / "cameras.bin"
    images_bin = input_dir / "images.bin"
    points3d_bin = input_dir / "points3D.bin"

    cameras_txt = input_dir / "cameras.txt"
    images_txt = input_dir / "images.txt"
    points3d_txt = input_dir / "points3D.txt"

    has_bin_pair = cameras_bin.is_file() and images_bin.is_file()
    has_txt_pair = cameras_txt.is_file() and images_txt.is_file()

    if not (has_bin_pair or has_txt_pair):
        raise FileNotFoundError(
            f"No valid COLMAP model found in {input_dir}: "
            "expected either (cameras.bin, images.bin) or (cameras.txt, images.txt)."
        )

    # Binary case: create a valid empty points3D.bin if missing
    if has_bin_pair and not points3d_bin.exists():
        with open(points3d_bin, "wb") as f:
            # Write number of points = 0 as uint64 (little endian)
            f.write(struct.pack("<Q", 0))

    # Text case: create a points3D.txt if missing
    if has_txt_pair and not points3d_txt.exists():
        header = (
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            "# Number of points: 0, mean track length: 0\n"
        )
        points3d_txt.write_text(header)


def save_colmap(
    self,
    output_dir: str,
    scale_factor: int = 1,
    same_intrinsics: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    cam_path = os.path.join(output_dir, "cameras.txt")
    img_path = os.path.join(output_dir, "images.txt")
    pts_path = os.path.join(output_dir, "points3D.txt")

    # Map original image ID to new name
    id_to_new_name = {image_id: f"{i:06d}.png" for i, image_id in enumerate(self.camera_ids)}

    # Write cameras and images
    with open(cam_path, "w") as cam_f, open(img_path, "w") as img_f:
        cam_total = 1 if same_intrinsics else len(self.camera_ids)

        cam_f.write("# Camera list with one line of data per camera:\n")
        cam_f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        cam_f.write(f"# Number of cameras: {cam_total}\n")

        img_f.write("# Image list with two lines of data per image:\n")
        img_f.write(
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        )
        img_f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        img_f.write(f"# Number of images: {len(self.camera_ids)}\n")

        written_cams = set()

        for idx, image_id in enumerate(self.camera_ids):
            img_obj = self.rec.images[image_id]
            cam_id = img_obj.camera_id
            K = self.K_dict[cam_id][scale_factor]
            width, height = self.imsize_dict[cam_id][scale_factor]
            if self.rotated_image[cam_id]:
                # Swap the resolution back to original for COLMAP. Intrinsics are before rotation, so no swap needed
                height, width = width, height  

            # Save camera intrinsics
            if not same_intrinsics or (same_intrinsics and idx == 0):
                if cam_id not in written_cams:
                    fx, fy = K[0, 0], K[1, 1]
                    cx, cy = K[0, 2], K[1, 2]
                    if abs(fx - fy) < 1e-6:
                        model = "SIMPLE_PINHOLE"
                        params = [fx, cx, cy]
                    else:
                        model = "PINHOLE"
                        params = [fx, fy, cx, cy]

                    cam_f.write(
                        f"{cam_id} {model} {width} {height} "
                        + " ".join(map(str, params))
                        + "\n"
                    )
                    written_cams.add(cam_id)

            # Save extrinsics with updated image name
            pose_cw = img_obj.cam_from_world() if callable(img_obj.cam_from_world) else img_obj.cam_from_world
            quat = pose_cw.rotation.quat
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            tx, ty, tz = pose_cw.translation
            new_name = id_to_new_name[image_id]

            img_f.write(
                f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {new_name}\n\n"
            )

    # Write or copy points3D
    original_pts = os.path.join(self.colmap_dir, "points3D.txt")
    if os.path.isfile(original_pts):
        shutil.copyfile(original_pts, pts_path)
    else:
        with open(pts_path, "w") as pts_f:
            pts_f.write("# 3D point list with one line of data per point:\n")
            pts_f.write(
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_ID)\n"
            )
            pts_f.write(f"# Number of points: {len(self.rec.points3D)}\n")

            for pid in sorted(self.rec.points3D):
                p = self.rec.points3D[pid]
                x, y, z = p.xyz
                r, g, b = p.color
                err = p.error
                track_str = " ".join(
                    f"{e.image_id} {e.point2D_idx}" for e in p.track.elements
                )
                pts_f.write(
                    f"{pid} {x} {y} {z} {r} {g} {b} {err} {track_str}\n"
                )
                