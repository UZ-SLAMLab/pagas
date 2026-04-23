# License: CC BY-NC 4.0 - see /LICENSE
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors


def project(points3D, intrinsics):
    """
    Pinhole camera projection.
    Args:
        points3D (torch.Tensor): A tensor of shape (..., 3) representing the 3D points.
            The last dimension contains the coordinates (X, Y, Z) for each point.
            The leading dimensions (...) can represent any number of batch dimensions.

        intrinsics (torch.Tensor): A tensor of shape (N, 4) or (N, 3, 3) representing the camera intrinsics.
            The first dimension N corresponds to the batch size.
            If shape is (N, 4), the second dimension contains the intrinsic parameters (fx, fy, cx, cy) for each camera.
            If shape is (N, 3, 3), it is the intrinsic matrix.
    Returns:
        torch.Tensor: A tensor of shape (..., 3) containing the projected 2D coordinates and depth.
            The last dimension contains the coordinates (x, y, Z), where:
            - x and y are the 2D projected coordinates.
            - Z is the depth.
            The leading dimensions match those of the input tensor points3D.
    """
    X, Y, Z = points3D.unbind(dim=-1)
    if intrinsics.shape[-1] == 4:
        fx, fy, cx, cy = intrinsics[:, None, None].unbind(dim=-1)
    elif intrinsics.shape[-1] == 3 and intrinsics.shape[-2] == 3:
        fx = intrinsics[:, 0, 0][:, None, None]
        fy = intrinsics[:, 1, 1][:, None, None]
        cx = intrinsics[:, 0, 2][:, None, None]
        cy = intrinsics[:, 1, 2][:, None, None]
    else:
        raise ValueError("Intrinsics tensor must have shape (N, 4) or (N, 3, 3)")
    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    coords = torch.stack([x, y, Z], dim=-1)
    return coords


def inv_project(depth, intrinsics):
    """
    Pinhole camera inverse-projection.
    Args:
        depth (torch.Tensor): A tensor of shape (..., H, W) representing the depth map.
            The last two dimensions correspond to the height (H) and width (W) of the depth map.
            The leading dimensions (...) can represent any number of batch dimensions.

        intrinsics (torch.Tensor): A tensor of shape (N, 4) or (N, 3, 3) representing the camera intrinsics.
            The first dimension N corresponds to the batch size.
            If shape is (N, 4), the second dimension contains the intrinsic parameters (fx, fy, cx, cy) for each camera.
            If shape is (N, 3, 3), it is the intrinsic matrix.
    Returns:
        torch.Tensor: A tensor of shape (..., H, W, 3) containing the 3D coordinates.
            The last dimension contains the coordinates (X, Y, Z), where:
            - X and Y are the 3D coordinates in the camera space.
            - Z is the depth value from the input.
            The leading dimensions match those of the input tensor depth.
    """
    ht, wd = depth.shape[-2:]
    if intrinsics.shape[-1] == 4:
        fx, fy, cx, cy = intrinsics[:, None, None].unbind(dim=-1)
    elif intrinsics.shape[-1] == 3 and intrinsics.shape[-2] == 3:
        fx = intrinsics[:, 0, 0][:, None, None]
        fy = intrinsics[:, 1, 1][:, None, None]
        cx = intrinsics[:, 0, 2][:, None, None]
        cy = intrinsics[:, 1, 2][:, None, None]
    else:
        raise ValueError("Intrinsics tensor must have shape (N, 4) or (N, 3, 3)")
    y, x = torch.meshgrid(
        torch.arange(ht).to(depth.device).float(),
        torch.arange(wd).to(depth.device).float(),
        indexing='ij'
    )
    x = x + 0.5  # center-aligned
    y = y + 0.5
    X = depth * ((x - cx) / fx)
    Y = depth * ((y - cy) / fy)
    Z = depth
    return torch.stack([X, Y, Z], dim=-1)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    From PyTorch3D.
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    From PyTorch3D.
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    From PyTorch3D.
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def depth_to_world_points(
    depth: Tensor, camtoworld: Tensor, K: Tensor, z_depth: bool = True
) -> Tensor:
    """Convert depth maps to 3D points

    Args:
        depth: Depth maps [..., H, W, 1]
        camtoworld: Camera-to-world transformation matrices [..., 4, 4]
        K: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        points: 3D points in the world coordinate system [..., H, W, 3]
    """
    assert depth.shape[-1] == 1, f"Invalid depth shape: {depth.shape}"
    assert camtoworld.shape[-2:] == (
        4,
        4,
    ), f"Invalid viewmats shape: {camtoworld.shape}"
    assert K.shape[-2:] == (3, 3), f"Invalid K shape: {K.shape}"
    assert (
        depth.shape[:-3] == camtoworld.shape[:-2] == K.shape[:-2]
    ), f"Shape mismatch! depth: {depth.shape}, viewmats: {camtoworld.shape}, K: {K.shape}"

    device = depth.device
    height, width = depth.shape[-3:-1]

    x, y = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="xy",
    )  # [H, W]
    x = x + 0.5  # center-aligned
    y = y + 0.5

    fx = K[..., 0, 0]  
    fy = K[..., 1, 1]  
    cx = K[..., 0, 2] 
    cy = K[..., 1, 2]  

    # camera directions in camera coordinates
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx[..., None, None]) / fx[..., None, None],
                (y - cy[..., None, None]) / fy[..., None, None],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [..., H, W, 3]

    # ray directions in world coordinates
    directions = torch.einsum(
        "...ij,...hwj->...hwi", camtoworld[..., :3, :3], camera_dirs
    )  # [..., H, W, 3]
    origins = camtoworld[..., :3, -1]  # [..., 3]

    if not z_depth:
        directions = F.normalize(directions, dim=-1)

    points = origins[..., None, None, :] + depth * directions
    return points


def depth_to_world_normal(
    depth: Tensor, camtoworld: Tensor, K: Tensor, z_depth: bool = True, method: str = "central"
) -> Tensor:
    """Convert depth maps to surface normal

    Args:
        depth: Depth maps [..., H, W, 1]
        camtoworld: Camera-to-world transformation matrices [..., 4, 4]
        K: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        normal: Surface normal in the world coordinate system [..., H, W, 3]
    """
    points = depth_to_world_points(depth, camtoworld, K, z_depth=z_depth)  # [..., H, W, 3]

    if method == "central":
        # Central differences
        dy, dx = torch.gradient(points, dim=(-3, -2))  # [..., H, W, 3]
    elif method == "forward":
        # Forward differences
        dx = points[..., :, 1:, :] - points[..., :, :-1, :]
        dy = points[..., 1:, :, :] - points[..., :-1, :, :]
        dx = torch.nn.functional.pad(dx, (0, 0, 0, 1))
        dy = torch.nn.functional.pad(dy, (0, 0, 0, 0, 0, 1))

    normal = - F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)

    return normal


def depth_to_world_rotation(
    depth: Tensor, camtoworld: Tensor, K: Tensor, z_depth: bool = True, method: str = "central"
) -> Tensor:
    """Convert depth maps to surface rotations.

    Args:
        depth: Depth maps [..., H, W, 1]
        camtoworld: Camera-to-world transformation matrices [..., 4, 4]
        K: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        rotations: Surface rotations in the world coordinate system in quaternions [..., H, W, 4] (wxyz)
    """
    points = depth_to_world_points(depth, camtoworld, K, z_depth=z_depth)  # [..., H, W, 3]

    if method == "central":
        # Central differences
        dy, dx = torch.gradient(points, dim=(-3, -2))  # [..., H, W, 3]
    elif method == "forward":
        # Forward differences
        dx = points[:, 1:, :] - points[:, :-1, :]
        dy = points[1:, :, :] - points[:-1, :, :]

    dx = dx / torch.norm(dx, dim=-1, keepdim=True)
    dy = dy / torch.norm(dy, dim=-1, keepdim=True)
    if method == "forward":
        dx, dy = dx[:-1, :], dy[:, :-1]
    normal = - F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)

    # Orthogonalize dx and dy to normal using Gram–Schmidt process
    dx = dx - normal * torch.sum(dx * normal, dim=-1, keepdim=True)
    dx = dx / torch.norm(dx, dim=-1, keepdim=True)
    dy = dy - normal * torch.sum(dy * normal, dim=-1, keepdim=True) - dx * torch.sum(dy * dx, dim=-1, keepdim=True)
    dy = dy / torch.norm(dy, dim=-1, keepdim=True)

    if method == "forward":
        dx = torch.nn.functional.pad(dx, (0, 0, 0, 1, 0, 1))
        dy = torch.nn.functional.pad(dy, (0, 0, 0, 1, 0, 1))
        normal = torch.nn.functional.pad(normal, (0, 0, 0, 1, 0, 1))
    # Get rotation
    R = torch.stack((dx, dy, normal), dim=-1)  # [..., H, W, 3, 3]
    q = matrix_to_quaternion(R)  # [..., H, W, 4] (wxyz)
    return q


def depth1D_to_3D(
    depth: Tensor,
    mask: Tensor,
    K: Tensor,
) -> Tensor:

    assert (
        depth.ndim == 1
    ), "depth must have 1 dimension: N (number of valid pixels)"
    device = depth.device

    # Create a grid of pixel coordinates
    batch_size, height, width = mask.shape[:3]
    y_indices, x_indices = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )  # [H, W]
    y_indices = y_indices.expand(batch_size, -1, -1).to(device) + 0.5  # center-aligned
    x_indices = x_indices.expand(batch_size, -1, -1).to(device) + 0.5
    # Extract the pixel coordinates where the mask is 1
    mask = mask.squeeze(-1)  # [B, H, W]
    x_masked = x_indices[mask]
    y_masked = y_indices[mask]
    # Project to 3D
    x_3d = (x_masked - K[..., 0, 2]) * depth / K[..., 0, 0]
    y_3d = (y_masked - K[..., 1, 2]) * depth / K[..., 1, 1]
    # Combine into a 3D point cloud
    means = torch.stack((x_3d, y_3d, depth), dim=-1)  # [N, 3]

    return means


def depth1D_to_2D(depth, mask):
    depth = depth.squeeze()
    mask = mask.squeeze()
    if len(depth.shape) == 2:
        rep_mask = mask.unsqueeze(0).repeat(depth.shape[0], 1, 1)
        depth_2D = torch.zeros_like(rep_mask, dtype=torch.float)
        depth_2D[rep_mask] = depth.flatten()
    else:
        depth_2D = torch.zeros_like(mask, dtype=torch.float)
        depth_2D[mask] = depth
    return depth_2D


def opacity1D_to_2D(opacity, mask):
    opacity_2D = torch.zeros_like(mask, dtype=torch.float)
    opacity_2D[mask] = opacity
    return opacity_2D


def scale1D_to_2D(scale, mask):
    """Convert a 1D scale tensor to a 2D tensor based on a mask."""
    scale_2D = torch.zeros((mask.shape[0], mask.shape[1], 3), dtype=torch.float).to(mask.device)
    scale_2D[mask] = scale
    return scale_2D


def knn(x: Tensor, K: int = 4) -> Tensor:
    """
    Compute the K-nearest neighbors distances for each point in a given tensor.
    Args:
        x (Tensor): A tensor of shape (N, D) where N is the number of points and D is the dimensionality of each point.
            The tensor should be on a device compatible with PyTorch (e.g., CPU or GPU).

        K (int, optional): The number of nearest neighbors to compute. Default is 4.
    Returns:
        Tensor: A tensor of shape (N, K) containing the distances to the K-nearest neighbors for each point.
            The distances are computed using the Euclidean metric.
    Note:
        - The function uses the `sklearn.neighbors.NearestNeighbors` class to compute the nearest neighbors.
        - The input tensor `x` is converted to a NumPy array for compatibility with scikit-learn, and the resulting distances
          are converted back to a PyTorch tensor.
        - The output tensor is returned on the same device as the input tensor `x`.
    """
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x.device)


def upsample_depth(image: np.ndarray, exclude_value: float) -> np.ndarray:
    is_torch = False
    if isinstance(image, torch.Tensor):
        is_torch = True
        device = image.device
        image = image.cpu().detach().numpy()
    # check image dimensions and data type
    assert image.ndim == 2 and image.dtype == np.float32
    h, w = image.shape[:2]
    h_up, w_up = h * 2, w * 2

    # generate upsampled image using nearest neighbor interpolation
    image_nn = np.zeros((h_up, w_up), dtype=np.float32)
    image_nn[::2, ::2] = image
    image_nn[::2, 1::2] = image
    image_nn[1::2, ::2] = image
    image_nn[1::2, 1::2] = image

    # pad image for linear interpolation
    sample_image = np.zeros((h_up + 2, w_up + 2), dtype=np.float32)
    sample_image[1:-1, 1:-1] = image_nn
    sample_image[0:1, :] = sample_image[1:2, :]
    sample_image[-1:, :] = sample_image[-2:-1, :]
    sample_image[:, 0:1] = sample_image[:, 1:2]
    sample_image[:, -1:] = sample_image[:, -2:-1]

    # extract subimages for upsampling
    image00 = sample_image[:h_up, :w_up]
    image01 = sample_image[:h_up, 1 : w_up + 1]
    image10 = sample_image[1 : h_up + 1, :w_up]
    image11 = sample_image[1 : h_up + 1, 1 : w_up + 1]

    # count valid pixels in full image
    image_count = np.zeros(image00.shape, dtype=np.float32)
    image_count += np.where(image00 == exclude_value, 0.0, 1.0)
    image_count += np.where(image01 == exclude_value, 0.0, 1.0)
    image_count += np.where(image10 == exclude_value, 0.0, 1.0)
    image_count += np.where(image11 == exclude_value, 0.0, 1.0)

    # generate upsampled image by averaging subimages
    image_up = (image00 + image01 + image10 + image11) / image_count.clip(min=1.0)
    return torch.from_numpy(image_up).to(device) if is_torch else image_up


def upsample_data(w_new, h_new, w_original, depth, opacities, mask, scales, cfg):
    with torch.no_grad():
        device = depth.device
        depth_init = upsample_depth(depth, 0.0).unsqueeze(0)

        opacity_init = None
        if not cfg.fix_opacity:
            opacity_init = opacity1D_to_2D(opacities, mask.squeeze())
            if len(opacity_init.shape) == 3:
                opacity_init = opacity_init.unsqueeze(0)
            else:
                opacity_init = opacity_init.unsqueeze(0).unsqueeze(0)
            opacity_init =  F.interpolate(opacity_init, size=(h_new, w_new), mode='nearest-exact').squeeze(0).detach()
        elif len(opacities.shape) == 3:
            opacity_init = None
            raise NotImplementedError("Add new parameter for multi-view-opacity and if fix_opacity=True, uses the value above")

        scale_init = None
        if scales is not None:
            if not cfg.fix_scale:
                scale_init = torch.log(torch.exp(scales) / (w_new.to(device) / w_original.to(device)))  # double resolution -> half scale
                if not cfg.depth_dependant_scale:
                    scale_init = scale1D_to_2D(scale_init, mask.squeeze())
                    scale_init =  F.interpolate(scale_init.unsqueeze(0).permute(0, 3, 1, 2), size=(h_new, w_new), mode='nearest-exact').squeeze(0).detach()
                    scale_init = scale_init.squeeze(0)  # [3, H, W]
                    scale_init = torch.log(torch.exp(scale_init) / (w_new.to(device) / w_original.to(device)))
            elif cfg.depth_dependant_scale:
                scale_init = scales / (w_new.to(device) / w_original.to(device))  # double resolution -> half scale

        return depth_init, opacity_init, scale_init


def get_pixel_dependant_scale(K: torch.tensor, mask: torch.tensor = None):
    """Relative (to Z-buffer depth) half side of the square of unprojecting a pixel to the 3D"""

    assert K.shape == (3, 3), f"Invalid K shape: {K.shape}"
    assert mask is None or len(mask.shape) == 2, "Invalid mask shape"
    device = K.device

    H, W = mask.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    u = (torch.arange(W).reshape(1, W) + 0.5).to(device)  # center-aligned
    v = (torch.arange(H).reshape(H, 1) + 0.5).to(device)

    euclidean_depth_relative = 1 / torch.sqrt(
        ((u - cx) / fx)**2 +
        ((v - cy) / fy)**2 +
        1
    )

    du = torch.tensor([1.0]).to(device)  # pixel width
    dv = torch.tensor([1.0]).to(device)  # pixel height

    half_side = euclidean_depth_relative * torch.sqrt(du * dv) / torch.sqrt(fx * fy) / 2.0

    if mask is not None:
        half_side = half_side[mask]
    else:
        half_side = half_side.reshape(-1)

    return half_side


def generate_binary_mask(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Generate a binary mask with ones in all pixels where a coordinate falls.

    If a coordinate does not fall exactly in the center of a pixel,
    the four closest pixels are set to 1.

    Args:
        coordinates (torch.Tensor): shape (B, H, W, 2)
            coordinates[..., 0] is x (width, column)
            coordinates[..., 1] is y (height, row)

    Returns:
        torch.Tensor: binary mask with shape (B, H, W)
    """
    B, H, W, _ = coordinates.shape
    device = coordinates.device

    # Output
    binary_mask = torch.zeros((B, H, W), device=device)

    # Flatten spatial dimensions
    N = H * W
    x = coordinates[..., 0].view(B, N)
    y = coordinates[..., 1].view(B, N)

    # Integer neighbor indices
    x0 = torch.floor(x).long()
    x1 = torch.ceil(x).long()
    y0 = torch.floor(y).long()
    y1 = torch.ceil(y).long()

    # Four neighbors per coordinate: (x0,y0), (x0,y1), (x1,y0), (x1,y1)
    xs = torch.stack([x0, x0, x1, x1], dim=-1)  # (B, N, 4)
    ys = torch.stack([y0, y1, y0, y1], dim=-1)  # (B, N, 4)

    # Batch indices with same shape
    b_idx = torch.arange(B, device=device)[:, None, None].expand(B, N, 4)

    # Keep only in bounds pixels, do not clamp
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)

    if valid.any():
        xs_valid = xs[valid]
        ys_valid = ys[valid]
        b_valid = b_idx[valid]
        binary_mask[b_valid, ys_valid, xs_valid] = 1.0

    return binary_mask


def get_unseen_areas_mask(camtoworld, context_camtoworld, input_depth_init, context_K, K):
    # Always input depth_init for the target view, not optimized to not lose the consistent 3D geometry
    depth_init_ = input_depth_init.squeeze(0).unsqueeze(-1)
    context_T = torch.cat([context_camtoworld.squeeze(0).inverse() @ camtoworld,],dim=0,)
    context_T = context_T.squeeze()
    context_K_ = context_K.squeeze()
    K_ = K.squeeze()

    if len(context_T.shape) == 2:
        context_T = context_T.unsqueeze(0)
        depth_init_ = depth_init_.unsqueeze(0)
        K_ = K_.unsqueeze(0)
        context_K_ = context_K_.unsqueeze(0)

    depth_init_ = depth_init_.repeat(context_T.shape[0], 1, 1, 1)
    K_ = K_.repeat(context_T.shape[0], 1, 1)
    points3d = depth_to_world_points(depth_init_, context_T, K_, z_depth=True)  # [M, H, W, 3]
    coor2D = project(points3d, context_K_)[..., :2]  # [M, H, W, 2]
    unseen_areas_mask = generate_binary_mask(coor2D)

    return unseen_areas_mask
