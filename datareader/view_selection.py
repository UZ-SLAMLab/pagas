# License: CC BY-NC 4.0 - see /LICENSE
import numpy as np
from typing import List, Dict, Tuple
import re


def write_views_file(
    indices: Dict[int, List[int]],
    image_names: List[str],
    output_path: str,
    set_first_index_to_0: bool = False
) -> None:
    """
    Write the views.cvg file using the indices dictionary and image_names list.
    
    Each pair of lines:
    - First line: target image name
    - Second line: context image names separated by ", "
    
    If set_first_index_to_0 is True:
        - The numeric part of the filenames is shifted so that the first image
          starts at 0, and the rest follow accordingly.
        - Indices in `indices` are not changed, only the names written.
          Example: 000003.jpg, 000004.jpg, ... -> 000000.jpg, 000001.jpg, ...
    """
    if set_first_index_to_0:
        first_name = image_names[0]
        m = re.search(r"(\d+)", first_name)
        if m is None:
            raise ValueError(f"Could not extract numeric index from filename: {first_name}")

        bias = int(m.group(1))          # e.g. 3 from "000003.jpg"
        pad_width = len(m.group(1))     # keep same zero-padding, e.g. 6

        output_image_names: List[str] = []
        for name in image_names:
            m2 = re.search(r"(\d+)", name)
            if m2 is None:
                raise ValueError(f"Could not extract numeric index from filename: {name}")

            original_num = int(m2.group(1))
            new_num = original_num - bias
            if new_num < 0:
                raise ValueError(
                    f"New index would be negative for filename {name} "
                    f"(original {original_num}, bias {bias})"
                )

            new_num_str = f"{new_num:0{pad_width}d}"
            start, end = m2.span(1)
            new_name = name[:start] + new_num_str + name[end:]
            output_image_names.append(new_name)
    else:
        output_image_names = image_names

    # Write file using indices as-is, but with possibly-renumbered names
    with open(output_path, 'w') as f:
        for i, target_name in enumerate(output_image_names):
            ctx_indices = indices.get(i, [])
            context_names = [output_image_names[j] for j in ctx_indices]
            f.write(f"{target_name}\n")
            f.write(", ".join(context_names) + "\n")


def camera_center(pose: np.ndarray) -> np.ndarray:
    """World‑space camera centre."""
    R = pose[:3, :3]
    t = pose[:3, 3]
    return -R.T @ t                    


def forward_axis(pose: np.ndarray) -> np.ndarray:
    """World‑space optical axis (points forward)."""
    return -pose[:3, 2]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine clipped to [0, 1] so opposite views score 0."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return max(0.0, float(np.dot(a, b)))


def exp_falloff(d: float, scale: float) -> float:
    """e^(−d/scale), high for nearby cameras, low for far ones."""
    return float(np.exp(-d / (scale + 1e-8)))


def median_baseline(centres: np.ndarray) -> float:
    """Robust rig scale used to cancel unknown units."""
    D = np.linalg.norm(centres[:, None] - centres[None, :], axis=-1)
    nz = D[D > 1e-8]
    return float(np.median(nz)) if nz.size else 1.0


def _sample_rays(K: np.ndarray, pose: np.ndarray,
                 im_size: Tuple[int, int], samp: int) -> np.ndarray:
    H, W = im_size
    us = np.linspace(0.5, W - 0.5, samp)  # center-aigned
    vs = np.linspace(0.5, H - 0.5, samp)
    uu, vv = np.meshgrid(us, vs)
    pix = np.stack([uu, vv, np.ones_like(uu)], axis=-1)

    dirs_cam = pix @ np.linalg.inv(K).T
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    return (dirs_cam @ pose[:3, :3].T).reshape(-1, 3)           


def _overlap_with_intrinsics(K_t: np.ndarray, pose_t: np.ndarray,
                             K_s: np.ndarray, pose_s: np.ndarray,
                             im_size_t: Tuple[int, int],
                             im_size_s: Tuple[int, int],
                             samp: int) -> float:
    """Fraction of target rays hitting the source image plane."""
    rays = _sample_rays(K_t, pose_t, im_size_t, samp)
    rays_s = rays @ pose_s[:3, :3]  # to source frame
    mask = rays_s[:, 2] > 1e-6
    if not np.any(mask):
        return 0.0
    rays_s = rays_s[mask]
    proj = (rays_s / rays_s[:, 2:3]) @ K_s.T
    u, v = proj[:, 0], proj[:, 1]
    H_s, W_s = im_size_s
    hit = (u >= 0) & (u < W_s) & (v >= 0) & (v < H_s)
    return float(hit.sum()) / float(rays.shape[0])


def _rank_one_with_intrinsics(idx_t: int,
                              Ks: List[np.ndarray],
                              poses: List[np.ndarray],
                              im_sizes: List[Tuple[int, int]],
                              w_overlap: float,
                              samp: int) -> List[int]:
    """Return indices sorted by score for one target (with intrinsics)."""
    K_t = Ks[idx_t]
    pose_t = poses[idx_t]
    im_size_t = im_sizes[idx_t]

    centres = np.vstack([camera_center(p) for p in poses])
    scale = median_baseline(centres)

    scores, indices = [], []
    for j, (K_s, pose_s, im_size_s) in enumerate(zip(Ks, poses, im_sizes)):
        if j == idx_t:
            continue
        overlap = _overlap_with_intrinsics(
            K_t, pose_t, K_s, pose_s, im_size_t, im_size_s, samp)
        dist = np.linalg.norm(camera_center(pose_s) - camera_center(pose_t))
        prox = exp_falloff(dist, scale)
        scores.append(w_overlap * overlap + (1.0 - w_overlap) * prox)
        indices.append(j)

    return [j for _, j in sorted(zip(scores, indices), reverse=True)]


def _rank_one_no_intrinsics(idx_t: int,
                            poses: List[np.ndarray],
                            w_ang: float,
                            scale: float) -> List[int]:
    """Return indices sorted by score for one target (no intrinsics)."""
    v_t = forward_axis(poses[idx_t])
    c_t = camera_center(poses[idx_t])

    scores, indices = [], []
    for j, pose_s in enumerate(poses):
        if j == idx_t:
            continue
        ang = cosine(v_t, forward_axis(pose_s))
        dist = np.linalg.norm(camera_center(pose_s) - c_t)
        prox = exp_falloff(dist, scale)
        scores.append(w_ang * ang + (1.0 - w_ang) * prox)
        indices.append(j)

    return [j for _, j in sorted(zip(scores, indices), reverse=True)]


def build_neighbor_dict(
    poses: List[np.ndarray],
    intrinsics: List[np.ndarray] | None = None,
    im_sizes: List[Tuple[int, int]] | None = None,
    n_neighbors: int = 3,
    w_overlap: float = 0.7,
    w_ang: float = 0.7,
    samp: int = 32,
) -> Dict[int, List[int]]:
    """
    For every pose return the zero‑based indices of its *n_neighbors*
    most compatible views.

    If *intrinsics* **and** *im_sizes* are provided (same length as poses),
    the intrinsics path is used, otherwise angular‑plus‑distance scoring
    is used.  Output dictionary keys and values are all zero based.
    """
    N = len(poses)

    use_intrinsics = intrinsics is not None and im_sizes is not None
    if use_intrinsics:
        if len(intrinsics) != N or len(im_sizes) != N:
            raise ValueError("intrinsics and im_sizes must match poses length")
    else:
        intrinsics = [None] * N
        im_sizes = [(1080, 1920)] * N   # dummy sizes (unused)

    centres = np.vstack([camera_center(p) for p in poses])
    scale = median_baseline(centres)

    neighbor_dict: Dict[int, List[int]] = {}

    for idx in range(N):
        if use_intrinsics:
            order = _rank_one_with_intrinsics(
                idx, intrinsics, poses, im_sizes, w_overlap, samp)
        else:
            order = _rank_one_no_intrinsics(
                idx, poses, w_ang, scale)

        neighbor_dict[idx] = order[:n_neighbors]  

    return neighbor_dict
