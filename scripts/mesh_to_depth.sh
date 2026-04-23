#!/bin/bash
# License: CC BY-NC 4.0 - see /LICENSE

data_dir=${1:-/.../DTU/scan24}
mesh_name=${2:-mesh_init.ply}
output_name=${3:-init}

python scripts/mesh_to_depth.py \
    --mesh_file=${data_dir}/${s}/${mesh_name} \
    --colmap_sparse_folder=${data_dir}/${s}/sparse \
    --output_depth_dir=${data_dir}/${s}/depth_${output_name} \
    --output_masks_dir=${data_dir}/${s}/masks_${output_name}

