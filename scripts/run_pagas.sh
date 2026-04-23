#!/bin/bash
# License: CC BY-NC 4.0 - see /LICENSE

# ------------------ Default values ------------------
data_dir="/.../DTU/scan24"
depth_folder="depth_init"
masks_name="masks"
scale_factors="2 1"
max_steps="100 100"
lr="1e-5 1e-5"
radius_thres="1.42 1.42"
depth_slices="100 100"
normal_reg="0."
num_context_views="10"
starting_view="0"

# Flags that will be optionally passed to train.py
viewer_arg=""
save_extra_arg=""
exposure_arg=""
use_alpha_weight_arg="--use_alpha_weight"

# ------------------ Parse named arguments ------------------
for arg in "$@"; do
    case $arg in
        --data_dir=*) data_dir="${arg#*=}" ;;
        --depth_folder=*) depth_folder="${arg#*=}" ;;
        --scale_factors=*) scale_factors="${arg#*=}" ;;
        --max_steps=*) max_steps="${arg#*=}" ;;
        --lr=*) lr="${arg#*=}" ;;
        --radius_thres=*) radius_thres="${arg#*=}" ;;
        --depth_slices=*) depth_slices="${arg#*=}" ;;
        --masks_name=*) masks_name="${arg#*=}" ;;
        --normal_reg=*) normal_reg="${arg#*=}" ;;
        --num_context_views=*) num_context_views="${arg#*=}" ;;
        --starting_view=*) starting_view="${arg#*=}" ;;
        --viewer) viewer_arg="--viewer" ;;
        --save_extra) save_extra_arg="--save_extra" ;;
        --exposure)
            exposure_arg="--exposure"
            use_alpha_weight_arg=""   
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# ------------------ Run training ------------------
python train.py \
    --data_dir="${data_dir}" \
    --result_dir="${data_dir}/results/${depth_folder}_pagas" \
    --depth_init_type="$depth_folder" \
    --masks_name="$masks_name" \
    --num_context_views="$num_context_views" \
    --starting_view="$starting_view" \
    --scale_factors $scale_factors \
    --max_steps $max_steps \
    --early_stop \
    --antialiased \
    --fix_opacity \
    --fix_scale \
    --depth_dependant_scale \
    --lr $lr \
    --normal_reg="$normal_reg" \
    --radius_thres $radius_thres \
    --depth_slices $depth_slices \
    $exposure_arg \
    $save_extra_arg \
    $use_alpha_weight_arg \
    --use_color_grad_weight \
    $viewer_arg
