# License: CC BY-NC 4.0 - see /LICENSE
from typing import Dict, Optional, Tuple
from typing_extensions import Literal

import torch
import torch.nn as nn
from torch import Tensor
from gsplat.rendering import rasterization

from .geo_ops import inv_project, knn, depth1D_to_3D, depth_to_world_rotation, depth_to_world_normal, get_pixel_dependant_scale
from datareader.image_ops import replace_zeros_with_nearest_neighbor


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


class AppModel(nn.Module):
    def __init__(self, num_images=1600, device="cuda"):  
        super().__init__()   
        self.appear_ab = nn.Parameter(torch.zeros(num_images, 2).to(device))
        self.optimizer = torch.optim.Adam([
                                {'params': self.appear_ab, 'lr': 0.001, "name": "appear_ab"},
                                ], betas=(0.9, 0.99))


def create_splats_with_optimizers(
    rgbs: torch.Tensor,
    depth_init: torch.Tensor,
    mask: torch.Tensor,
    K: torch.Tensor,
    scale_init: Optional[torch.Tensor] = None,
    opacity_init: Optional[torch.Tensor] = None,
    init_opacity: float = 0.999,
    init_scale: float = 1.0,
    sh_degree: int = 3,
    batch_size: int = 1,
    device: str = "cuda",
    fix_scale: bool = False,
    fix_opacity: bool = False,
    type_of_depth_to_optimize: str = "depth",
    scales_to_optimize_and_type_of_gaussians: str = "three_3d",
    rotation_type: Literal["random", "from_depth", "optimized"] = "random",
    depth_dependant_scale: bool = False,
    lr: float = 1e-4,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:

    assert batch_size == 1, "Splats initialization only supports batch size 1"

    if depth_dependant_scale:
        assert scales_to_optimize_and_type_of_gaussians in ["one_3d", "one_2d"], "a single global scale can only be used with 1 scale gaussians"

    if len(mask.shape) != 2:
        mask = mask.squeeze()
    mask = mask.unsqueeze(0)

    depth = depth_init[mask]  # [N, 1]
    rgbs = rgbs[mask.squeeze(0)]  # [N, 3]

    # Scale initialization
    if depth_dependant_scale and fix_scale:
        scales = get_pixel_dependant_scale(K.squeeze(0), mask.squeeze(0)).unsqueeze(-1).repeat(1, 3)
    elif scale_init is None:
        points = inv_project(depth_init, K)  # [B, H, W, 3]
        points = points[mask]  # [N, 3]
        dist2_avg = ((knn(points, 4)[:, 1:] ** 2).mean(dim=-1).float())  # [N,].  This assumes batch 1
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1)  # [N, 1]  for "one_3d", "one_2d"
        if scales_to_optimize_and_type_of_gaussians in ["two_3d", "two_2d"]:
            scales = scales.repeat(1, 2)  # [N, 2]
        elif scales_to_optimize_and_type_of_gaussians in ["three_3d"]:
            scales = scales.repeat(1, 3)  # [N, 3]
        elif depth_dependant_scale:
            scales = torch.log(torch.mean(get_pixel_dependant_scale(K.squeeze(0), mask.squeeze(0)).unsqueeze(-1)))
    else:
        if depth_dependant_scale:
            if fix_scale:
                scales = scale_init
            else:
                scales = torch.log(scale_init)
        else:
            if fix_scale:
                scales = replace_zeros_with_nearest_neighbor(scale_init)[mask]
            else:
                scales = torch.log(replace_zeros_with_nearest_neighbor(scale_init)[mask.repeat(3, 1, 1)]).reshape(-1, 3)  # [N, 3]

    # Opacity initialization
    N = rgbs.shape[0]
    if fix_opacity:
            opacities = torch.full((N,), init_opacity)  # [N,]
    else:
        if opacity_init is None:
            init_opacity = min(0.999, init_opacity)
            opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
        else:
            opacities = torch.logit(replace_zeros_with_nearest_neighbor(opacity_init)[mask])  # [N, 1]

    params = []  # name, value, lr
    if type_of_depth_to_optimize == "inverse_depth":
        params.append(("depth", torch.nn.Parameter(1 / depth), lr))
    elif type_of_depth_to_optimize == "log_depth":
        params.append(("depth", torch.nn.Parameter(torch.log(depth)), lr))
    else:
        params.append(("depth", torch.nn.Parameter(depth), lr))
    depth = None

    if not fix_scale:
        params.append(("scales", torch.nn.Parameter(scales), 5e-3))
        scales = None
    elif not depth_dependant_scale:
        if scale_init is None:
            # fix the scale during the whole training
            if scales_to_optimize_and_type_of_gaussians == "two_3d":
                second_col = scales[:, 1].unsqueeze(1)
                scales = torch.cat((scales, second_col), dim=1)
            elif scales_to_optimize_and_type_of_gaussians == "two_2d":
                zeros_col = torch.zeros(scales.shape[0], 1)
                scales = torch.cat((scales, zeros_col), dim=1)
            elif scales_to_optimize_and_type_of_gaussians == "one_3d":
                scales = scales.repeat(1, 3)
            elif scales_to_optimize_and_type_of_gaussians == "one_2d":
                zeros_col = torch.zeros(scales.size(0), 1)
                scales = torch.cat((scales.expand(-1, 2), zeros_col), dim=1)
        scales = scales.to(device)

    if not fix_opacity:
        params.append(("opacities", torch.nn.Parameter(opacities), lr))
        opacities = None
    else:
        opacities = opacities.to(device)

    # Rotation initialization
    if rotation_type == "optimized":
        quats = depth_to_world_rotation(depth_init.squeeze(0).unsqueeze(-1), torch.eye(4).to(device), K.squeeze(0), method="central").squeeze(0)  # [H, W, 4]
        quats = quats[mask.squeeze()]
        params.append(("quats", torch.nn.Parameter(quats), lr))
        quats = None
    elif rotation_type == "random":
        quats = torch.rand((N, 4)).to(device)  # [N, 4]
    elif rotation_type == "from_depth":
        quats = None

    # Color is SH coefficients
    if sh_degree != -1:
        color = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        color[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(color[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(color[:, 1:, :]), 2.5e-3 / 20))
        color = None
    else:
        color = rgb_to_sh(rgbs).unsqueeze(1)
        color = color.to(device)

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr, "name": name}],
        )
        for name, _, lr in params
    }
    return splats, optimizers, depth, scales, opacities, color, quats


def rasterize_splats(
    self,
    camtoworld: Tensor,
    context_camtoworld: Tensor,
    K: Tensor,
    context_K: Tensor,
    all_mask: Optional[Tensor] = None,
    scale_lambda_at_step: float = 1.0,
    depth2D: Optional[Tensor] = None,
    rotated_images: Tensor = torch.tensor([]),
    radius_thres: float = 1.5,
    depth_thres: float = 1e10,
    app_model: Optional[AppModel] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:

    assert (camtoworld.shape[0] == 1), "Splats initialization only supports batch size 1"
    assert kwargs['render_mode'] in {"RGB", "RGB+ED"}, "Invalid render_mode. Choose either RGB or RGB+ED."
    assert len(all_mask.shape) == 3, "all_mask should be of shape [B, H, W]"

    # Project depth to 3D to initialize the means. We extract the valid coordinates using the mask.
    depth = self.splats["depth"]  # [N, 1]
    if self.cfg.type_of_depth_to_optimize == "inverse_depth":
        depth = 1 / depth
    elif self.cfg.type_of_depth_to_optimize == "log_depth":
        depth = torch.exp(depth)
    means = depth1D_to_3D(depth, all_mask[0].unsqueeze(0), K)

    # Rotations
    if self.cfg.rotation_type == "from_depth":
        quats = depth_to_world_rotation(depth2D.unsqueeze(-1), torch.eye(4).to(self.device), K.squeeze(0), method=self.cfg.depth_to_normal_method).squeeze(0)  # [H, W, 4]
        quats = quats[all_mask[0].squeeze()]
    elif self.cfg.rotation_type == "optimized":
        quats = self.splats["quats"]  # [N, 4]  w, x, y, z
    elif self.cfg.rotation_type == "random":
        quats = self.quats

    # Scales
    scale_factor = depth if self.cfg.depth_dependant_scale else 1.
    if not self.cfg.fix_scale:
        scales = torch.exp(self.splats["scales"]) * scale_factor
        if self.cfg.depth_dependant_scale:
            scales = scales.unsqueeze(-1).repeat(1, 3)
        elif self.cfg.scales_to_optimize_and_type_of_gaussians == "two_3d":
            first_col = scales[:, 0].unsqueeze(1)
            scales = torch.cat((first_col, scales), dim=1)
        elif self.cfg.scales_to_optimize_and_type_of_gaussians == "two_2d":
            zeros_col = torch.zeros(scales.shape[0], 1).to(self.device)
            scales = torch.cat((scales, zeros_col), dim=1)
        elif self.cfg.scales_to_optimize_and_type_of_gaussians == "one_3d":
            scales = scales.repeat(1, 3)
        elif self.cfg.scales_to_optimize_and_type_of_gaussians == "one_2d":
            zeros_col = torch.zeros(scales.size(0), 1, device=self.device)
            scales = torch.cat((scales.expand(-1, 2), zeros_col), dim=1)
    else:
        if self.cfg.depth_dependant_scale:
            scales = self.scales * scale_factor.unsqueeze(-1) * scale_lambda_at_step
        else:
            scales = self.scales * scale_factor

    # Opacities
    if not self.cfg.fix_opacity:
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
    else:
        opacities = self.opacities  # [N,]

    _ = kwargs.pop("image_id", None)

    # Color
    if self.color is None:
        color = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
    else:
        color = self.color

    height, width = all_mask.shape[-2:]

    # Camera calibration
    all_camtoworld = torch.cat(
        [
            torch.eye(4).unsqueeze(0).to(self.device),
            camtoworld.inverse() @ context_camtoworld.squeeze(0),
        ], dim=0,
    )
    all_K = torch.cat([K, context_K.squeeze(0)], dim=0)

    rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

    render_color, render_alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=color,
        viewmats=torch.linalg.inv(all_camtoworld),  # [C, 4, 4]
        Ks=all_K,  # [C, 3, 3]
        width=width,
        height=height,
        depth_thres=depth_thres,
        radius_thres=radius_thres,
        packed=self.cfg.packed,
        absgrad=False,
        rasterize_mode=rasterize_mode,
        **kwargs,
    )

    # Compute normal from optimized depth
    T = torch.eye(4).to(self.device)
    if rotated_images.numel() > 0:
        if rotated_images[0, 0].item() == 0:
            rotation90 = torch.tensor([
                [0., -1., 0., 0.],
                [1.,  0., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]
            ]).to(self.device)
            T = rotation90
    normal = depth_to_world_normal(depth2D.unsqueeze(-1), T, K.squeeze(0), method=self.cfg.depth_to_normal_method).squeeze(0)

    # Exposure compensation model
    if app_model is not None:
        render_color = render_color / render_alpha.clamp(min=1e-10)
        appear_ab = app_model.appear_ab
        app_image = torch.exp(appear_ab[:, 0])[:, None, None, None] * render_color + appear_ab[:, 1][:, None, None, None]
        app_image[~all_mask] = 0.0
    else:
        app_image = None

    return render_color, normal, render_alpha, app_image
