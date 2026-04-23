#!/usr/bin/env python3
# License: CC BY-NC 4.0 - see /LICENSE
"""
Compute TSDF parameters (voxel_size, depth_trunc) from COLMAP poses and mesh_res.

Usage examples:
  # Auto detect under current folder
  python compute_tsdf_params.py --mesh_res 1024

  # Explicit COLMAP directory
  python compute_tsdf_params.py --colmap_dir /path/to/scan/sparse/0 --mesh_res 1024

Output format:
  [tsdf_params]
  VOXEL_SIZE=0.003337030 DEPTH_TRUNC=3.417119
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

from pycolmap import Reconstruction


def quat_xyzw_to_R(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = map(float, q_xyzw)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)],
    ], dtype=np.float64)


def rigid3d_to_4x4(Tcw_obj) -> np.ndarray:
    M = getattr(Tcw_obj, "matrix", None)
    if callable(M):
        M = np.asarray(M(), dtype=np.float64)
        if M.shape == (3, 4):
            M = np.vstack([M, np.array([0, 0, 0, 1], dtype=np.float64)])
        elif M.shape != (4, 4):
            raise ValueError(f"Unexpected Tcw matrix shape: {M.shape}")
        return M

    rot = getattr(Tcw_obj, "rotation", None)
    trans = getattr(Tcw_obj, "translation", None)
    if rot is None or trans is None:
        raise TypeError("Rigid3d lacks pose fields")

    q = getattr(rot, "quat", None)
    if q is None:
        q = getattr(rot, "coeffs", None)
    if q is None:
        q = rot
    q = np.asarray(q, dtype=np.float64).reshape(4,)
    qw, qx, qy, qz = q
    R = quat_xyzw_to_R(np.array([qx, qy, qz, qw], dtype=np.float64))
    t = np.asarray(trans, dtype=np.float64).reshape(3,)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_camtoworld(colmap_dir: Path) -> np.ndarray:
    rec = Reconstruction(str(colmap_dir))
    if len(rec.images) == 0:
        raise ValueError(f"No images in COLMAP reconstruction at {colmap_dir}")

    names, Twc_list = [], []
    for _, img in rec.images.items():
        pose_cw = getattr(img, "cam_from_world", None)
        if pose_cw is not None:
            Tcw_obj = pose_cw() if callable(pose_cw) else pose_cw
            Tcw = rigid3d_to_4x4(Tcw_obj)
        else:
            qvec = getattr(img, "qvec", None)
            tvec = getattr(img, "tvec", None)
            if qvec is None or tvec is None:
                raise ValueError("Missing pose on image")
            qw, qx, qy, qz = map(float, qvec)
            R = quat_xyzw_to_R(np.array([qx, qy, qz, qw], dtype=np.float64))
            Tcw = np.eye(4, dtype=np.float64)
            Tcw[:3, :3] = R
            Tcw[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3,)

        Twc = np.linalg.inv(Tcw)
        names.append(img.name)
        Twc_list.append(Twc)

    order = np.argsort(np.asarray(names, dtype=object))
    Twc_sorted = [Twc_list[i] for i in order]
    return np.stack(Twc_sorted, axis=0).astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description="Compute voxel_size and depth_trunc from COLMAP poses.")
    ap.add_argument("--mesh_res", type=int, required=True, help="Target mesh resolution. voxel_size = depth_trunc / mesh_res.")
    ap.add_argument("--colmap_dir", type=str, default="", help="Optional path to COLMAP sparse directory. If not provided, tries ./sparse/0 then ./sparse.")
    args = ap.parse_args()

    if args.mesh_res <= 0:
        print("mesh_res must be > 0", file=sys.stderr)
        sys.exit(2)

    if args.colmap_dir:
        colmap_dir = Path(args.colmap_dir).resolve()
    else:
        cwd = Path(os.getcwd())
        candidates = [cwd / "sparse" / "0", cwd / "sparse"]
        colmap_dir = next((p for p in candidates if p.is_dir()), None)

    if colmap_dir is None or not colmap_dir.is_dir():
        print("COLMAP directory not found. Provide --colmap_dir or ensure ./sparse/0 or ./sparse exists", file=sys.stderr)
        sys.exit(1)

    camtoworld = load_camtoworld(colmap_dir)
    cam_pos = camtoworld[:, :3, 3]
    bb_min = cam_pos.min(0)
    bb_max = cam_pos.max(0)
    center = 0.5 * (bb_min + bb_max)
    dists = np.linalg.norm(cam_pos - center, axis=1)
    radius = float(dists.max())
    depth_trunc = 2.0 * radius
    voxel_size = depth_trunc / float(args.mesh_res)

    print(f"\033[1;34m[TSDF_PARAMS] VOXEL_SIZE={voxel_size:.9f} DEPTH_TRUNC={depth_trunc:.6f}\033[0m")


if __name__ == "__main__":
    main()
