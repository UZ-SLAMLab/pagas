# License: CC BY-NC 4.0 - see /LICENSE
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from typing_extensions import Literal


import torch
from torch.utils.tensorboard import SummaryWriter

import tyro
import yaml
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
try:
    import rerun as rr
except Exception:
    rr = None

from datareader.colmap import Dataset, Parser
from datareader.utils import data_to_cuda

from pagas.splat_ops import create_splats_with_optimizers, rasterize_splats, AppModel
from pagas.geo_ops import upsample_data, get_unseen_areas_mask
from pagas.loss import get_loss
from pagas.utils import valid_params_verification, preprocess_optimized_data, animate_rocket

from scripts.seed import set_random_seed, worker_init_fn
from scripts.log import log, save_data


@dataclass
class Config:
    # Path to the Colmap style dataset
    data_dir: str = ""
    # Path to ground truth depth to use in the evaluation. Relative path to data_dir. If not provided but viewer or tensorboard is True, the depth_init_type is used.
    gt_depth_folder: Optional[str] = ""
    # Context views file, relative to data_dir.
    views_file: str = "views.cfg"
    # Number of context views. If -1, use all closest views.
    num_context_views: int = 10
    # Downsample factor for the dataset. If several, coarse-to-fine training is performed. Enter from lower (e.g. 1) to higher (e.g. 16) data factors
    scale_factors: List[Literal[16, 8, 4, 2, 1]] = field(default_factory=lambda: [2, 1])
    # Directory to save results
    result_dir: str = "results/pagas"
    # Train on a specific view only
    target_view: Optional[int] = None
    # Starting training view. Skip the ones before.
    starting_view: int = 0

    # Normalize the world space
    normalize_world_space: bool = True

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1

    # Number of training steps per pyramid scale. If several, indicates the number of steps on each pyramid level, starting with the coarsest one.
    max_steps: List[int] = field(default_factory=lambda: [200, 100])

    # Initialization depth strategy
    depth_init_type: str = "depth_init"
    # Masks folder name
    masks_name: str = "masks"
    # Degree of spherical harmonics. -1 means color is not optimized, is fixed
    sh_degree: int = -1
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 20
    # Initial opacity of GS. Valid range: (0, 1]
    init_opa: float = 1.0
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss at each pyramid level, starting with the coarsest one.
    ssim_lambda: List[float] = field(default_factory=lambda: [0.2])
    # Type of depth to optimize
    type_of_depth_to_optimize: Literal["depth", "inverse_depth", "log_depth"] = "depth"
    # Number of scales to optimize and type of gaussians. 2D gaussians have the third scale fixed to 0. one_3d = sphere, one_2d = circular disk, two_3d = disk with thickness, two_2d = disk
    scales_to_optimize_and_type_of_gaussians: Literal["one_3d", "one_2d", "two_3d", "two_2d", "three_3d"] = "one_3d"
    # Make scale of each dependant of its depth, and only a single scale parameter is optimized for all Gaussians. Only works with 1 scale parameter gaussians, i.e. scales_to_optimize_and_type_of_gaussians=["one_3d", "one_2d"]
    depth_dependant_scale: bool = False
    # All Gaussian scales are multiplied by this lambda that linearly converges to 1 at the relative step specified by scale_lambda_relative_step_end. Requires arguments --fix_scale and --depth_dependant_scale
    scale_lambda: List[float] = field(default_factory=lambda: [1.0])
    # Range (0, 1], 1 meaning the end of the pyramid level
    scale_lambda_relative_step_end: float = 1.0
    # Gaussians rotation type: random, derived from optimized depth, or optimized individually. If optimized, rotations is initialized with normals derived from depth_init
    rotation_type: Literal["random", "from_depth", "optimized"] = "random"
    # Method to get the normal from depth
    depth_to_normal_method: Literal["central", "forward"] = "central"
    # Depth learning rate
    lr: List[float] = field(default_factory=lambda: [1e-5])
    # If True, detect a plateau in the loss and stop iterating.
    early_stop: bool = False

    # Near plane clipping distance
    near_plane: float = 0.0
    # Far plane clipping distance
    far_plane: float = float('inf')
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Fix gaussians scale
    fix_scale: bool = False
    # Fix gaussians opacity. Set opacity to 1.
    fix_opacity: bool = False

    # Depth and radius threshold for rasterizer, indicating firt and last value for each scale.
    radius_thres: List[Tuple[float, float]] = field(default_factory=lambda: [(1.42, 1.42)])  # 0.71 is the distance to the corner of the pixel from the center
    # Depth threshold for rasterizer, indicating first and last value for each scale.
    depth_thres: Optional[List[Tuple[float, float]]] = None
    # Alternatively to use depth_thres, you can provide the depth_slices number (indicating firt and last value for each scale) that is used to divide the input normalized sparse depth range by it in order to get depth_thres
    depth_slices: Optional[List[Tuple[int, int]]] = field(default_factory=lambda: [(20, 20)])

    # Normal regularization for each level of the pyramid, starting with the coarsest one.
    normal_reg: List[float] = field(default_factory=lambda: [0.0])

    # Use render alpha as weight in the photometric loss
    use_alpha_weight: bool = False
    # Use color gradient based weight
    use_color_grad_weight: bool = False

    # Learn per-iamge exposure compensation
    exposure: bool = False

    # Log information to tensorboard/viewer every these steps. -1 means no log
    log_every: int = 10
    # Save training images every these steps. -1 means no log
    log_image_every: int = 10
    # Show rerun viewer
    viewer: bool = False
    # Log in tensorboard
    tensorboard: bool = False
    # Save extra data: refined normals and colorized depth and normals.
    save_extra: bool = False


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, cfg: Config
    ) -> None:
        set_random_seed(42)            

        self.cfg = cfg
        self.device = f"cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu": print("WARNING: CPU only, this will be very slow!")
        self.create_out_folders()

        # Load data: Training data should contain initial points and color.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            masks_name=cfg.masks_name,
            views_file=cfg.views_file,
            gt_depth_folder=cfg.depth_init_type if cfg.viewer or cfg.tensorboard and cfg.gt_depth_folder == "" else cfg.gt_depth_folder,
            scale_factors=cfg.scale_factors,
            normalize=cfg.normalize_world_space,
            depth_init_type=cfg.depth_init_type,
            num_context_views=cfg.num_context_views,
            target_view=cfg.target_view,
            starting_view=cfg.starting_view,
            use_color_grad_weight=cfg.use_color_grad_weight,
            result_dir=cfg.result_dir,
            colmap_results_dir=self.colmap_results_dir,
        )
        self.trainset = Dataset(self.parser)


    def create_out_folders(self) -> None:
        self.render_dir = f"{self.cfg.result_dir}"
        self.color_dir = f"{self.cfg.result_dir}/images"
        self.depth_np_dir = f"{self.cfg.result_dir}/depth"
        self.depth_color_dir = f"{self.cfg.result_dir}/depth_color"
        self.normal_np_dir = f"{self.cfg.result_dir}/normal"
        self.normal_color_dir = f"{self.cfg.result_dir}/normal_color"
        self.masks_dir = f"{self.cfg.result_dir}/masks"
        self.colmap_results_dir = f"{self.cfg.result_dir}/sparse/0"

        # Always-created directories
        dirs = [
            self.cfg.result_dir,
            self.render_dir,
            self.color_dir,
            self.depth_np_dir,
            self.masks_dir,
            self.colmap_results_dir,
        ]

        # Optional extra data directories
        if getattr(self.cfg, "save_extra", False):
            dirs.extend([
                self.depth_color_dir,
                self.normal_np_dir,
                self.normal_color_dir,
            ])

        for dir in dirs:
            os.makedirs(dir, exist_ok=True)


    def train(self):
        device = self.device
        self.cfg = valid_params_verification(self.cfg)
        cfg = self.cfg
        normalization_scale = self.parser.normalization_scale.item()

        # Dump cfg
        with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=8,
            worker_init_fn=worker_init_fn,
            persistent_workers=False,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

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

        # Views loop
        with progress:
            outer = progress.add_task("[magenta]Views", total=len(trainloader))
            log_line = ""

            for view_i in range(len(trainloader)):
                try:
                    data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    data = next(trainloader_iter)

                camtoworld = data["camtoworld"].to(device)  # [B, 4, 4]
                context_camtoworld = data["context_camtoworld"].to(device)  # [B, M, 4, 4]
                image_id = data["image_id"].to(device)  # [B,]
                imsize = data["imsize"]  # {B, 2}
                rotated_images = data["rotated_images"].to(device)  # [M + 1]
                opacity_init = None
                scale_init = None
                total_steps = 0

                app_model = None
                if cfg.exposure:
                    app_model = AppModel(num_images=(context_camtoworld.shape[1] + 1), device=device)
                    app_model.train()
                    app_model.to(device)

                self.writer = None
                if cfg.tensorboard:
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb/{current_time}")

                if cfg.viewer and rr is not None:
                    rr.init("PAGaS", spawn=True)
                    rr.log_file_from_path("rerun.rbl")

                scale_factors = sorted(cfg.scale_factors, reverse=True)
                for scale_idx, scale_factor in enumerate(scale_factors):
                    if log_line:
                        console.file.write("\033[F\033[K")
                    log_line = f"[bold red]:fire: Refining view {view_i}"
                    console.print(log_line)                   

                    ssim_lambda = cfg.ssim_lambda[scale_idx]
                    normal_reg = cfg.normal_reg[scale_idx]
                    lr = cfg.lr[scale_idx]
                    scale_lambda = cfg.scale_lambda[scale_idx]
                
                    K, context_K, all_pixels, all_mask, gt_depth, gt_normal, inv_color_grad_weight, input_depth_init = \
                        data_to_cuda(data, scale_factor, cfg, device)

                    self.splats, self.optimizers, self.depth, self.scales, self.opacities, self.color, self.quats = create_splats_with_optimizers(
                        rgbs=all_pixels[0],
                        depth_init=input_depth_init if scale_idx == 0 else depth_init,
                        mask=all_mask[0],
                        K=K,
                        scale_init=scale_init,  # scales of previous scale
                        opacity_init=opacity_init,  # opacity of previous scale
                        init_opacity=cfg.init_opa,  # starting opacity value
                        init_scale=cfg.init_scale,
                        sh_degree=cfg.sh_degree,
                        batch_size=cfg.batch_size,
                        device=self.device,
                        fix_scale=cfg.fix_scale,
                        fix_opacity=cfg.fix_opacity,
                        type_of_depth_to_optimize=cfg.type_of_depth_to_optimize,
                        scales_to_optimize_and_type_of_gaussians=cfg.scales_to_optimize_and_type_of_gaussians,
                        rotation_type=cfg.rotation_type,
                        depth_dependant_scale=cfg.depth_dependant_scale,
                        lr=lr,
                    )

                    if app_model is not None:
                        self.optimizers["app_model"] = app_model.optimizer
 
                    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizers["depth"], patience=10)]
                    if cfg.exposure:
                        schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(app_model.optimizer, patience=10))

                    # Depth threshold setup
                    scale_steps = cfg.max_steps[scale_idx]
                    radius_thres_start, radius_thres_end = cfg.radius_thres[scale_idx]
                    radius_thres_delta = (radius_thres_end - radius_thres_start) / (scale_steps - 1)
                    if cfg.depth_thres is not None:                    
                        depth_thres_start, depth_thres_end = cfg.depth_thres[scale_idx]      
                        depth_thres_start *= normalization_scale
                        depth_thres_end *= normalization_scale 
                    else:
                        input_depth_init_range = (input_depth_init.max() - input_depth_init[input_depth_init > 0.].min()).item()
                        depth_thres_start = input_depth_init_range / cfg.depth_slices[scale_idx][0]
                        depth_thres_end = input_depth_init_range / cfg.depth_slices[scale_idx][1]
                    depth_thres_delta = (depth_thres_end - depth_thres_start) / (scale_steps - 1)

                    # Pixeles of context views that are not seen in target
                    unseen_areas_mask = get_unseen_areas_mask(camtoworld, context_camtoworld, input_depth_init, context_K, K)

                    inner = progress.add_task(f"×{scale_factor}", total=scale_steps)

                    # Training loop
                    for step in range(scale_steps):                        
                        if cfg.early_stop:
                            if self.optimizers["depth"].param_groups[0]['lr'] < 1e-7:
                                break

                        sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree) if cfg.sh_degree > 0 else 0
                        depth, opacities = preprocess_optimized_data(self, all_mask[0])

                        scale_lambda_at_step = 1.0
                        if self.cfg.depth_dependant_scale and self.cfg.fix_scale:
                            scale_lambda_at_step = (scale_lambda - 1) * (1 - min(1., step / (scale_steps * cfg.scale_lambda_relative_step_end))) + 1

                        radius_thres = radius_thres_delta * step + radius_thres_start
                        depth_thres = depth_thres_delta * step + depth_thres_start

                        # Rasterization
                        (
                            color,
                            normal,
                            render_alpha,
                            app_image
                        ) = rasterize_splats(
                            self=self,
                            camtoworld=camtoworld,
                            context_camtoworld=context_camtoworld,
                            K=K,
                            context_K=context_K,
                            sh_degree=sh_degree_to_use,
                            near_plane=cfg.near_plane,
                            far_plane=cfg.far_plane,
                            image_id=image_id,
                            render_mode="RGB",
                            all_mask=all_mask,
                            scale_lambda_at_step=scale_lambda_at_step,
                            depth2D=depth,
                            rotated_images=rotated_images,
                            radius_thres=radius_thres,
                            depth_thres=depth_thres,
                            app_model=app_model,
                        )

                        if unseen_areas_mask is not None:
                            all_pixels[1:] *= unseen_areas_mask.unsqueeze(-1)                            

                        (
                            loss,
                            l1loss,
                            ssimloss,
                            normal_smooth_loss_value,
                        ) = get_loss(
                            color,
                            all_pixels,
                            all_mask,
                            inv_color_grad_weight,
                            ssim_lambda,
                            normal,
                            normal_reg,
                            render_alpha,
                            cfg.use_alpha_weight,
                            app_image,
                            )
                        loss.backward()
                        
                        log(
                            self.writer,
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
                            scale_idx,
                            self.optimizers["depth"].param_groups[0]['lr'],
                            )

                        # Optimizer
                        for optimizer in self.optimizers.values():
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                        for scheduler in schedulers:
                            scheduler.step(loss)

                        total_steps += 1
                        progress.advance(inner)

                    progress.remove_task(inner)

                    # Upsample to initialize next scale
                    if scale_idx < len(scale_factors) - 1:
                        next_factor = scale_factors[scale_idx + 1]
                        w_new, h_new = imsize[next_factor]
                        w_original = imsize[scale_factor][0]
                        depth_init, opacity_init, scale_init = upsample_data(w_new, h_new, w_original, depth, opacities, all_mask[0], self.splats["scales"].data if "scales" in self.splats else None, cfg)
                
                save_data(
                    all_pixels[0],
                    depth,
                    normalization_scale,
                    normal,
                    all_mask[0],
                    self.color_dir,
                    self.depth_np_dir,
                    self.depth_color_dir,
                    self.normal_np_dir,
                    self.normal_color_dir,
                    self.masks_dir,
                    image_id,
                    rotated_image=data["rotated_image"].item(),
                    save_extra=cfg.save_extra
                )
                progress.advance(outer)

            if log_line:
                console.file.write("\033[F\033[K")        

        torch.cuda.empty_cache()
        animate_rocket(f"✅ Scene {os.path.basename(cfg.data_dir)} refined by PAGaS 😎")


def main(cfg: Config):
    runner = Runner(cfg)
    runner.train()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
