#!/usr/bin/env python3
# License: CC BY-NC 4.0 - see /LICENSE
"""
Multi-view geometric consistency filter for per-view depth maps
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datareader.colmap import get_indices

import numpy as np
import cv2
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)


console = Console()


# ---------- COLMAP TXT loaders ----------

def parse_cameras_txt(path: str) -> Dict[int, dict]:
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            parts = line.split()
            # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cams[cam_id] = dict(model=model, width=width, height=height, params=params)
    return cams


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    # q = [qw, qx, qy, qz]
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)


def parse_images_txt(path: str) -> Dict[int, dict]:
    imgs = {}
    with open(path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f]  # Keep empty lines

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith('#'):
            i += 1
            continue

        parts = s.split()
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        cam_id = int(parts[8])
        name = parts[9]

        R = qvec2rotmat(qvec)
        t = tvec
        imgs[image_id] = dict(R=R, t=t, cam_id=cam_id, name=name)

        # consume the points2D line (may be empty)
        i += 1
        if i < len(lines):
            i += 1

    return imgs


def build_intrinsics(cam: dict) -> Tuple[np.ndarray, np.ndarray]:
    model = cam['model'].upper()
    p = cam['params']
    if model in ['SIMPLE_PINHOLE']:  # fx=fy=f, cx, cy
        f, cx, cy = p
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]], dtype=np.float64)
        D = np.zeros(5)
    elif model in ['PINHOLE']:  # fx, fy, cx, cy
        fx, fy, cx, cy = p
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
        D = np.zeros(5)
    elif model in ['SIMPLE_RADIAL', 'RADIAL', 'OPENCV', 'OPENCV_FISHEYE']:
        # Expect undistorted images for consistency filtering
        # We ignore distortion in this script
        if model == 'SIMPLE_RADIAL':
            f, cx, cy, _k1 = p
            K = np.array([[f, 0, cx],
                          [0, f, cy],
                          [0, 0, 1]], dtype=np.float64)
        elif model == 'RADIAL':
            f, cx, cy, _k1, _k2 = p
            K = np.array([[f, 0, cx],
                          [0, f, cy],
                          [0, 0, 1]], dtype=np.float64)
        elif model == 'OPENCV':
            fx, fy, cx, cy, *_ = p
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float64)
        else:  # OPENCV_FISHEYE
            fx, fy, cx, cy, *_ = p
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float64)
        D = np.zeros(5)
    else:
        raise ValueError(f"Unsupported camera model: {model}")
    return K, D


# ---------- Utility ----------

def load_depth_npz(path_npz: str) -> np.ndarray:
    data = np.load(path_npz)
    for key in ['depth', 'arr_0', 'depth_map']:
        if key in data:
            return data[key].astype(np.float32)
    # Fallback: first array
    for k in data.files:
        return data[k].astype(np.float32)
    raise ValueError(f"No depth array found in {path_npz}")


def load_mask_optional(path_png: str, shape_hw: Tuple[int, int]) -> np.ndarray:
    if not os.path.isfile(path_png):
        return np.ones(shape_hw, dtype=np.uint8)
    m = cv2.imread(path_png, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.ones(shape_hw, dtype=np.uint8)
    m = (m > 0).astype(np.uint8)
    if m.shape != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m


def bilinear_sample(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    depth: HxW
    u, v: arrays of same shape in pixel coordinates
    returns sampled depths, invalid samples as 0
    """
    H, W = depth.shape
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    valid = (u0 >= 0) & (v0 >= 0) & (u1 < W) & (v1 < H)
    out = np.zeros_like(u, dtype=np.float32)
    if not np.any(valid):
        return out

    ua = u[valid] - u0[valid]
    va = v[valid] - v0[valid]

    d00 = depth[v0[valid], u0[valid]]
    d01 = depth[v1[valid], u0[valid]]
    d10 = depth[v0[valid], u1[valid]]
    d11 = depth[v1[valid], u1[valid]]

    out_valid = (1 - ua) * ((1 - va) * d00 + va * d01) + ua * ((1 - va) * d10 + va * d11)
    out[valid] = out_valid.astype(np.float32)
    return out


def refine_depth_for_view(
    idx_ref: int,
    images_list: List[dict],
    cams_map: Dict[int, dict],
    K_list: List[np.ndarray],
    Kinv_list: List[np.ndarray],
    depth_dir: Path,
    masks_dir: Path,
    out_dir_depth: Path,
    out_dir_mask: Path,
    neighbor_ids: List[int],
    t_abs: float,
    t_rel: float
):
    img_ref = images_list[idx_ref]
    name_ref = img_ref['name']
    cam_ref = cams_map[img_ref['cam_id']]
    K_ref = K_list[idx_ref]
    Kinv_ref = Kinv_list[idx_ref]
    H, W = cam_ref['height'], cam_ref['width']

    depth_path = depth_dir / (Path(name_ref).stem + ".npz")
    if not depth_path.is_file():
        print(f"[WARN] Missing depth for {name_ref}")
        return

    D_ref = load_depth_npz(str(depth_path))  # HxW, float32
    if D_ref.shape != (H, W):
        D_ref = cv2.resize(D_ref, (W, H), interpolation=cv2.INTER_NEAREST)

    M_ref = load_mask_optional(str((masks_dir / Path(name_ref).with_suffix(".png"))), (H, W)).astype(np.uint8)
    valid_ref = (D_ref > 0) & (M_ref > 0)

    # Precompute pixel grid and rays in camera ref
    u_coords = np.arange(W, dtype=np.float32)
    v_coords = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)  # HxW

    ones = np.ones_like(uu)
    pix_ref = np.stack([uu, vv, ones], axis=-1).reshape(-1, 3).T  # 3xN
    Dv = D_ref.reshape(-1)
    valid_flat = valid_ref.reshape(-1)

    # Rays in ref camera
    rays_ref = Kinv_ref @ pix_ref  # 3xN
    X_ref_cam = rays_ref * Dv  # 3xN

    # Transform ref camera coords to world: X_w = R_ref^T (X_ref_cam - t_ref)
    R_ref = img_ref['R']; t_ref = img_ref['t']
    X_w = (R_ref.T @ (X_ref_cam - t_ref.reshape(3,1)))  # 3xN

    # Accumulators
    support = np.zeros_like(Dv, dtype=np.int32)
    z_ref_candidates = []  # list of arrays, each candidate z_ref from a neighbor

    # Reference thresholds
    delta = np.maximum(t_abs, t_rel * Dv)

    for idx_nbr in neighbor_ids:
        img_nb = images_list[idx_nbr]
        name_nb = img_nb['name']
        cam_nb = cams_map[img_nb['cam_id']]
        K_nb = K_list[idx_nbr]
        Kinv_nb = Kinv_list[idx_nbr]
        Hn, Wn = cam_nb['height'], cam_nb['width']

        # Load neighbor depth and mask
        depth_nb_path = depth_dir / (Path(name_nb).stem + ".npz")
        if not depth_nb_path.is_file():
            continue
        D_nb = load_depth_npz(str(depth_nb_path))
        if D_nb.shape != (Hn, Wn):
            D_nb = cv2.resize(D_nb, (Wn, Hn), interpolation=cv2.INTER_NEAREST)
        M_nb = load_mask_optional(str((masks_dir / Path(name_nb).with_suffix(".png"))), (Hn, Wn)).astype(np.uint8)

        # World to neighbor camera: X_nb = R_nb X_w + t_nb
        R_nb = img_nb['R']; t_nb = img_nb['t']
        X_nb = (R_nb @ X_w) + t_nb.reshape(3,1)   # 3xN

        z_nb = X_nb[2, :]
        eps = 1e-6
        valid_z = z_nb > eps

        # Project to neighbor pixels
        p_nb = K_nb @ X_nb
        u_nb = (p_nb[0, :] / (z_nb + eps))
        v_nb = (p_nb[1, :] / (z_nb + eps))

        # Sample neighbor depth at projected coords
        d_nb_sampled = bilinear_sample(D_nb, u_nb, v_nb)
        m_nb_sampled = bilinear_sample(M_nb.astype(np.float32), u_nb, v_nb) > 0.5

        # Depth agreement test in neighbor frame
        agree = valid_z & m_nb_sampled & (d_nb_sampled > 0)
        agree &= np.abs(d_nb_sampled - z_nb) <= delta

        # For agreeing pixels, take neighbor depth and convert back to ref camera z
        # Reconstruct world point from neighbor sampled depth and pixel
        # X_nb_samp = d_nb_sampled * Kinv_nb @ [u,v,1]
        rays_nb = Kinv_nb @ np.vstack([u_nb, v_nb, np.ones_like(u_nb)])
        X_nb_samp = rays_nb * d_nb_sampled
        X_w_from_nb = (R_nb.T @ (X_nb_samp - t_nb.reshape(3,1)))  # 3xN

        X_ref_from_nb = (R_ref @ X_w_from_nb) + t_ref.reshape(3,1)
        z_ref_from_nb = X_ref_from_nb[2, :].astype(np.float32)

        # Keep only where agree
        z_ref_cand = np.where(agree, z_ref_from_nb, np.nan)
        z_ref_candidates.append(z_ref_cand)
        support += agree.astype(np.int32)

    # Decide valid refined pixels: support >= k and was valid_ref
    z_stack = None
    if len(z_ref_candidates) > 0:
        # Stack candidates, upcast to float64 for safer reductions
        z_stack = np.stack(z_ref_candidates, axis=0).astype(np.float64)  # M x N
        # Include the original depth as a candidate too
        z_stack = np.concatenate([z_stack, Dv[None, :].astype(np.float64)], axis=0)

        # Optionally treat non positive or non finite values as invalid
        valid_entries = np.isfinite(z_stack) & (z_stack > 0)
        z_stack[~valid_entries] = np.nan

        # Median over candidates ignoring NaNs
        valid_any = np.any(np.isfinite(z_stack), axis=0)
        z_med = np.nanmedian(z_stack, axis=0)  # float64

        # Where all candidates were invalid, fall back to original depth
        z_med = np.where(valid_any, z_med, Dv.astype(np.float64))
        z_med = z_med.astype(np.float32)
    else:
        z_med = Dv.copy()

    # Final mask: original valid and enough multi-view support
    mask_refined = (valid_flat) & (support >= args.k_support)
    D_out = np.where(mask_refined, z_med, 0.0).reshape(H, W).astype(np.float32)
    M_out = mask_refined.reshape(H, W).astype(np.uint8)

    out_path_depth = out_dir_depth / (Path(name_ref).stem + ".npz")
    out_path_mask = out_dir_mask / (Path(name_ref).stem + ".png")
    np.savez_compressed(str(out_path_depth), D_out)
    cv2.imwrite(str(out_path_mask), (M_out * 255).astype(np.uint8))


def main(args):    
    scene = Path(args.data_folder)
    sparse = scene / args.sparse_folder
    depth_dir = scene / args.depth_folder
    masks_dir = scene / args.masks_folder
    out_dir_depth = scene / 'depth_consistent'
    out_dir_mask = scene / 'mask_consistent'

    out_dir_depth.mkdir(parents=True, exist_ok=True)
    out_dir_mask.mkdir(parents=True, exist_ok=True)

    cams = parse_cameras_txt(str(sparse / "cameras.txt"))
    imgs_map = parse_images_txt(str(sparse / "images.txt"))

    # Build ordered list of images by image_id to align arrays
    image_ids = sorted(imgs_map.keys())
    images_list = [imgs_map[iid] for iid in image_ids]

    # Intrinsics per image index
    K_list, Kinv_list = [], []
    for im in images_list:
        K, _ = build_intrinsics(cams[im['cam_id']])
        K_list.append(K)
        Kinv_list.append(np.linalg.inv(K))

    # Neighbor lists
    indices = get_indices(scene / args.views_file, args.neighbors)

    # Progress bars setup
    console = Console()
    columns = [
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total}", style="cyan"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    progress = Progress(*columns, console=console)

    with progress:
        task = progress.add_task("[magenta]Views", total=len(images_list))
        
        console.file.write("\033[F\033[K")
        log_line = f"[bold red]:fire: Applying stereo depth consitency filtering"
        console.print(log_line) 

        for i in range(len(images_list)):   
            progress.advance(task)
            
            # Classic geometric consistency + median consensus
            neighbor_ids = indices[i]
            refine_depth_for_view(
                i, images_list, cams, K_list, Kinv_list,
                depth_dir, masks_dir, out_dir_depth,
                out_dir_mask, neighbor_ids,
                t_abs=args.t_abs, t_rel=args.t_rel
            )
    
        console.file.write("\033[F\033[K") 
        console.print("\033[1;34m[STEREO CONSISTENCY] Done applying stereo depth consistency filtering\033[0m", highlight=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--sparse_folder", type=str, default="sparse/0")
    parser.add_argument("--depth_folder", type=str, default="depth")
    parser.add_argument("--masks_folder", type=str, default="masks")
    parser.add_argument("--views_file", type=str, default="views.cfg")
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--k_support", type=int, default=3)
    parser.add_argument("--t_abs", type=float, default=0.01, help="absolute depth tolerance (meters)")
    parser.add_argument("--t_rel", type=float, default=0.01, help="relative tolerance (fraction of depth). 0.01 = 1%")
    
    args = parser.parse_args()
    main(args)
