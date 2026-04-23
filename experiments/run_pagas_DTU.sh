#!/bin/bash
# License: CC BY-NC 4.0 - see /LICENSE

data_dir=${1:-/.../DTU}
depth_folder=${2:-depth_2dgs}
mask_folder=${3:-masks}

scans="scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122"


for s in $scans; do
    python train.py \
        --data_dir=${data_dir}/$s \
        --result_dir=${data_dir}/$s/results/${depth_folder}_pagas \
        --depth_init_type=$depth_folder \
        --masks_name=$mask_folder \
        --scale_factors 2 1 \
        --max_steps 100 100 \
        --early_stop \
        --antialiased \
        --fix_opacity \
        --fix_scale \
        --depth_dependant_scale \
        --lr 1e-5 1e-5 \
        --normal_reg=0.0 \
        --radius_thres 1.42 1.42 \
        --depth_slices 20 20 \
        --use_alpha_weight \
        --use_color_grad_weight
done
