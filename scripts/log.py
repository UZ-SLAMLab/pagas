# License: CC BY-NC 4.0 - see /LICENSE
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
try:
    import rerun as rr
except Exception:
    rr = None

from pagas.geo_ops import inv_project
from .metrics import EVAL_METRICS, eval_depth, eval_normal


def log(
    writer,
    step,
    total_steps,
    view_i,
    cfg,
    loss,
    l1loss,
    ssimloss,
    normal_smooth_loss_value,
    normal_reg,
    all_mask,
    K,
    gt_depth,
    depth,
    normalization_scale,
    normal,
    gt_normal,
    block_i,
    lr,
    ):
    with torch.no_grad():       
        if cfg.log_every != -1 and (cfg.tensorboard or cfg.viewer) and view_i == 0:
            mask = all_mask[0].unsqueeze(0)     
            rr.set_time("step", sequence=total_steps) if cfg.viewer else None
            if cfg.log_every > 0 and (step % cfg.log_every == 0 or step == cfg.max_steps[block_i] - 1):
                mem = torch.cuda.max_memory_allocated() / 1024**3
                if cfg.tensorboard:
                    writer.add_scalar(f"loss", loss.item(), total_steps)
                    writer.add_scalar(f"l1loss", l1loss.item(), total_steps)
                    writer.add_scalar(f"ssimloss", ssimloss.item(), total_steps) if ssimloss > 0.0 else None
                    writer.add_scalar(f"normal_smooth_loss", normal_smooth_loss_value.item(), total_steps) if normal_reg > 0.0 else None
                    writer.add_scalar(f"num_GS", mask.sum().item(), total_steps)
                    writer.add_scalar(f"mem", mem, total_steps)
                    writer.add_scalar(f"lr", lr, total_steps)

                depth[~mask.squeeze()] = 0.0
                normal[~mask.squeeze()] = 0.0
                if gt_depth is not None and cfg.tensorboard:
                    mask_gt_depth = (gt_depth.squeeze(0) > 0.)
                    mask_est_depth = (depth > 0.)
                    # Only for the evaluation, we combine the segmentation mask with the invalid gt depth pixels
                    eval_mask = mask.squeeze() * mask_gt_depth * mask_est_depth
                    metric_values = eval_depth(depth / normalization_scale, gt_depth.squeeze(0) / normalization_scale, eval_mask)
                    for metric_name, metric_value in zip(EVAL_METRICS, metric_values):
                        writer.add_scalar(f"depth_metrics/{metric_name}", metric_value, total_steps)
                    normal_error = eval_normal(normal, gt_normal, eval_mask)
                    writer.add_scalar(f"normal_error", normal_error, total_steps)

            if (
                cfg.log_image_every != -1
                and (step % cfg.log_image_every == 0 or step == cfg.max_steps[block_i] - 1)
            ):
                canvas = - torch.concat([normal.detach().cpu()], dim=1)
                canvas = (canvas * 0.5 + 0.5).numpy()
                canvas = (canvas * 255).astype(np.uint8)
                canvas[~mask.cpu().numpy().squeeze()] = 0
                if cfg.tensorboard:
                    writer.add_image(f"Normal Render", canvas.transpose(2, 0, 1), total_steps)
                if cfg.viewer:
                    rr.log("NormalRender", rr.Image(canvas))

                # Point cloud
                if cfg.viewer:
                    rr.log(
                        "K",
                        rr.Pinhole(
                            resolution=depth.shape,
                            image_from_camera=K.detach().cpu().numpy(),
                            image_plane_distance=0.3
                        ),
                    )

                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ]) @ np.array([
                        [1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]
                    ])

                    rr.log(
                        "K",
                        rr.Transform3D(
                            translation=np.array([0., 0., 0.]),
                            mat3x3=rotation_matrix,
                            from_parent=False,
                        ),
                    )

                    # Colors
                    depth_color = np.array([0, 120, 255, 255], dtype=np.uint8)    
                    gt_depth_color = np.array([255, 40, 40, 255], dtype=np.uint8) 

                    # Predicted (normalized) depth point cloud
                    depth_pc = inv_project(depth.unsqueeze(0), K)
                    depth_pc = depth_pc.squeeze().detach().cpu().numpy()
                    depth_pc = depth_pc[depth_pc[..., -1] > 0.]
                    depth_pc = (rotation_matrix @ depth_pc.T).T
                    rr.log("depth", rr.Points3D(depth_pc, colors=depth_color))

                    # Ground truth (normalized) depth point cloud
                    depth_pc = inv_project(gt_depth.unsqueeze(0), K)
                    depth_pc = depth_pc.squeeze().detach().cpu().numpy()
                    depth_pc = depth_pc[depth_pc[..., -1] > 0.]
                    depth_pc = (rotation_matrix @ depth_pc.T).T
                    rr.log("gt_depth", rr.Points3D(depth_pc, colors=gt_depth_color))


def save_data(
        color, 
        depth, 
        normalization_scale, 
        normal, 
        mask, 
        color_dir, 
        depth_np_dir, 
        depth_color_dir, 
        normal_np_dir, 
        normal_color_dir, 
        masks_dir, 
        image_id, 
        rotated_image=False, 
        save_extra=False
    ):
    # Save color images
    color *= 255
    color = color.detach().cpu().numpy()
    color = Image.fromarray(color.astype(np.uint8))

    depth[~mask] = 0.0
    depth = depth.detach().cpu().numpy()

    if save_extra:
        # save depth colorized
        canvas = depth[None]
        depth_min, depth_max = np.min(canvas[canvas > 0]), np.max(canvas)
        canvas = (canvas - depth_min) / (depth_max - depth_min)
        canvas[~mask.unsqueeze(0).cpu().numpy()] = depth_min
        cmap = plt.colormaps['Spectral']
        canvas = cmap(canvas.transpose(1, 2, 0)[:, :, 0])[:, :, :3]
        canvas = (canvas * 255.).astype(np.uint8)
        canvas[~mask.cpu().numpy()] = 255
        depth_color = Image.fromarray(canvas)

        # save normal
        normal = - normal.detach().cpu().numpy()
        normal[~mask.cpu().numpy()] = 0.
        canvas = (normal * 0.5 + 0.5)
        canvas = np.clip(canvas, 0, 1)
        canvas = (canvas * 255).astype(np.uint8)
        canvas[~mask.cpu().numpy()] = 255
        color_normal = Image.fromarray(canvas)

    # save mask as binary image (0 or 255)
    mask = mask.cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask) 

    if rotated_image:
        color = color.rotate(-90, expand=True)
        depth = np.rot90(depth, -1)
        mask = mask.rotate(-90, expand=True)
        if save_extra:
            depth_color = depth_color.rotate(-90, expand=True)
            normal = np.rot90(normal, -1)
            color_normal = color_normal.rotate(-90, expand=True)
        
    color.save(os.path.join(color_dir, f"{int(image_id):06d}.png"))
    np.savez(os.path.join(depth_np_dir, f"{int(image_id):06d}.npz"), (depth / normalization_scale).astype(np.float32))
    mask.save(os.path.join(masks_dir, f"{int(image_id):06d}.png"))

    if save_extra:
        depth_color.save(os.path.join(depth_color_dir, f"{int(image_id):06d}.png"))
        np.savez(os.path.join(normal_np_dir, f"{int(image_id):06d}.npz"), - normal.astype(np.float32))
        color_normal.save(os.path.join(normal_color_dir, f"{int(image_id):06d}.png"))
        