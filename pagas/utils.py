# License: CC BY-NC 4.0 - see /LICENSE
import os
import torch

from typing import List, Tuple
from rich.console import Console

from .geo_ops import depth1D_to_2D

import random
import time
from rich.console import Console
from rich.live import Live
from rich.text import Text


console = Console()

ROCKET     = "🚀"
EXPLOSIONS = ["💥", "✨", "🎆", "🎇"]
COLORS     = ["yellow", "magenta", "cyan", "green", "blue", "red"]


def animate_rocket(text: str = "✅ Done!", width: int = 30, step_delay: float = 0.04) -> None:
    """ Animate a rocket launch in the console."""
    with Live(console=console,
              transient=True,
              refresh_per_second=int(1 / step_delay)) as live:
        for x in range(width):
            live.update(Text(" " * x + ROCKET, style=random.choice(COLORS)))
            time.sleep(step_delay)

        for frame in EXPLOSIONS:
            live.update(Text(" " * width + frame, style=random.choice(COLORS)))
            time.sleep(0.2)

    console.log(Text(text, style="bold green"))


def valid_params_verification(cfg):
    assert (cfg.batch_size == 1), "Batch size should be 1 since we are optimizing one view at a time"

    if len(cfg.max_steps) !=1:
        assert (len(cfg.max_steps) == len(cfg.scale_factors)), "The number of max steps should be equal to the number of scale factors or just 1."
    else:
        cfg.max_steps = cfg.max_steps * len(cfg.scale_factors)

    if len(cfg.normal_reg) !=1:
        assert (len(cfg.normal_reg) == len(cfg.scale_factors)), "The number of normal_reg items should be equal to the number of scale factors or just 1."
    else:
        cfg.normal_reg = cfg.normal_reg * len(cfg.scale_factors)

    if len(cfg.ssim_lambda) !=1:
        assert (len(cfg.ssim_lambda) == len(cfg.scale_factors)), "The number of ssim_lambda items should be equal to the number of scale factors or just 1."
    else:
        cfg.ssim_lambda = cfg.ssim_lambda * len(cfg.scale_factors)

    if len(cfg.lr) !=1:
        assert (len(cfg.lr) == len(cfg.scale_factors)), "The number of lr items should be equal to the number of scale factors or just 1. Now is {}.".format(len(cfg.lr))
    else:
        cfg.lr = cfg.lr * len(cfg.scale_factors)

    if len(cfg.scale_lambda) !=1:
        assert (len(cfg.scale_lambda) == len(cfg.scale_factors)), "The number of scale_lambda items should be equal to the number of scale factors or just 1."
    else:
        cfg.scale_lambda = cfg.scale_lambda * len(cfg.scale_factors)

    assert cfg.scale_lambda_relative_step_end > 0.0 and cfg.scale_lambda_relative_step_end <= 1.0, "scale_lambda_relative_step_end must be in range (0, 1])"

    if len(cfg.radius_thres) !=1:
        assert (len(cfg.radius_thres) == len(cfg.scale_factors)), "The number of radius_thres tupples of length 2 should be equal to the number of scale factors or just 1."
    else:
        cfg.radius_thres = cfg.radius_thres * len(cfg.scale_factors)

    if cfg.depth_thres is not None:
        if len(cfg.depth_thres) !=1:
            assert (len(cfg.depth_thres) == len(cfg.scale_factors)), "The number of depth_thres tupples of length 2 should be equal to the number of scale factors or just 1."
        else:
            cfg.depth_thres = cfg.depth_thres * len(cfg.scale_factors)

    if cfg.depth_slices is not None:
        if len(cfg.depth_slices) !=1:
            assert (len(cfg.depth_slices) == len(cfg.scale_factors)), "The number of depth_slices tupples of length 2 should be equal to the number of scale factors or just 1."
        else:
            cfg.depth_slices = cfg.depth_slices * len(cfg.scale_factors)

    return cfg


def preprocess_optimized_data(self, mask):
    depth1D = self.splats["depth"]  # [N] depth of the target view
    if self.cfg.type_of_depth_to_optimize == "inverse_depth":
        depth1D = 1 / depth1D
    if self.cfg.type_of_depth_to_optimize == "log_depth":
        depth1D = torch.exp(depth1D)
    depth = depth1D_to_2D(depth1D, mask.squeeze())
    if not self.cfg.fix_opacity:
        opacities = torch.abs(torch.sigmoid(self.splats["opacities"]))
    else:
        opacities = self.opacities

    return depth, opacities
