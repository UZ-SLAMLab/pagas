# License: CC BY-NC 4.0 - see /LICENSE
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
import torch
import cv2


def replace_zeros_with_nearest_neighbor(depth_map):
    """
    Replace zero depth values with the closest non-zero depth value for each item in the batch.
    Args:
        depth_map (torch.Tensor or np.ndarray): Depth map with shape B x H x W.
    Returns:
        torch.Tensor or np.ndarray: Depth map with no zeros, same shape as input.
    """
    is_torch_tensor = isinstance(depth_map, torch.Tensor)
    if is_torch_tensor:
        depth_map_np = depth_map.cpu().numpy()
    else:
        depth_map_np = depth_map
    filled_depth_map_np = np.zeros_like(depth_map_np)
    mask = (depth_map_np != 0)
    _, (batch_indices, idx_x, idx_y) = distance_transform_edt(~mask, return_indices=True)
    filled_depth_map_np = depth_map_np.copy()
    filled_depth_map_np[~mask] = depth_map_np[batch_indices[~mask], idx_x[~mask], idx_y[~mask]]
    if is_torch_tensor:
        filled_depth_map = torch.from_numpy(filled_depth_map_np).to(depth_map.device)
    else:
        filled_depth_map = filled_depth_map_np
    return filled_depth_map


def compute_smoothing_weight_color(color: np.ndarray, mask: np.ndarray, min_color_grad: float = 0.02, max_color_grad: float = 0.1, min_weight: float = 0.0) -> np.ndarray:
    """Compute smoothing weights for batched color and mask inputs using vectorized operations.

    Args:
        color: A numpy array of shape (batch_size, height, width, channels) with values in [0, 1].
        mask: A numpy array of shape (batch_size, height, width) with boolean values.
        min_color_grad: Low threshold for the color gradient.
        max_color_grad: High threshold for the color gradient.
        min_weight: Minimum weight value.

    Returns:
        A numpy array of shape (batch_size, height, width) with the computed weights.
    """
    assert color.max() <= 1.0 and color.min() >= 0.0

    # Detect border pixels in the mask
    erosion_size = 5
    structuring_element = np.ones((erosion_size, erosion_size), dtype=bool)
    mask_erode = np.array([binary_erosion(m, structure=structuring_element) for m in mask])

    # Compute gradients
    grad_img_x = np.zeros(color.shape[:3], dtype=np.float32)
    grad_img_y = np.zeros(color.shape[:3], dtype=np.float32)

    grad_img_x[:, :, :-1] = np.mean(np.abs(color[:, :, :-1] - color[:, :, 1:]), axis=3)
    grad_img_y[:, :-1, :] = np.mean(np.abs(color[:, :-1, :] - color[:, 1:, :]), axis=3)

    # Select maximum dot product and normalize to range [0.0, 1.0]
    grad_img = np.maximum(grad_img_x, grad_img_y)

    weight = np.clip(grad_img, min_color_grad, max_color_grad)
    weight = (weight - min_color_grad) / (max_color_grad - min_color_grad)  # Normalize to [0, 1]
    weight = 1 - weight
    weight = np.clip(weight, min_weight, 1.0)
    weight[~mask_erode] = 0.0

    return weight
