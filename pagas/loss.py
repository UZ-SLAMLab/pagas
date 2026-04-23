# License: CC BY-NC 4.0 - see /LICENSE
import torch
from fused_ssim import fused_ssim


def normal_smoothness_loss(normal, weight=None, mask=None):
    """Computes the smoothness loss for a normal image.
    """
    dotx = - torch.sum(normal[:, :-1] * normal[:, 1:], dim=-1)  # range [-1,1] -1 good, 1 bad
    doty = - torch.sum(normal[:-1, :] * normal[1:, :], dim=-1)

    dotx = dotx * 0.5 + 0.5
    doty = doty * 0.5 + 0.5

    dotx = dotx[1:-1, 1:]
    doty = doty[1:, 1:-1]

    dot = (dotx + doty) / 2.
    dot[~mask[1:-1, 1:-1]] = 0.

    if weight is not None:
        dot = dot * weight[1:-1, 1:-1]

    return dot.mean()


def get_loss(
    color,
    all_pixels,
    all_mask,
    inv_color_grad_weight,
    ssim_lambda,
    normal,
    normal_reg,
    render_alpha,
    use_alpha_weight,
    app_image,
    ):
    # Photometric SSIM loss
    ssimloss = 0.0
    if ssim_lambda > 0.0:
        ssimloss = 1.0 - fused_ssim(color.permute(0, 3, 1, 2), all_pixels.permute(0, 3, 1, 2), padding="valid")
    
    # Photometric L1 loss
    l1loss_weight = 1.0
    if use_alpha_weight:
        l1loss_weight *= render_alpha
    if inv_color_grad_weight is not None:
        l1loss_weight *= (1.0 - inv_color_grad_weight).unsqueeze(-1)

    # Exposure compensation
    if app_image is not None and ssimloss < 0.5:
        assert app_image is not None and ssim_lambda > 0.0, "ssim_lambda > 0.0 should be provided when app_image is not None"
        l1loss = torch.mean(torch.abs(app_image - all_pixels) * l1loss_weight)
    else:
        l1loss = torch.mean(torch.abs(color - all_pixels) * l1loss_weight)

    loss = l1loss * (1.0 - ssim_lambda) + ssimloss * ssim_lambda

    # Normal smoothness loss
    normal_smooth_loss_value = 0.0
    if normal_reg > 0.0:
        normal_smooth_loss_value = normal_reg * normal_smoothness_loss(normal, weight=(inv_color_grad_weight[0] if inv_color_grad_weight is not None else None), mask=all_mask[0])
        loss += normal_smooth_loss_value

    return loss, l1loss, ssimloss, normal_smooth_loss_value
