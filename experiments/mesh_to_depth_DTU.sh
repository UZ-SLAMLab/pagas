#!/bin/bash
# License: CC BY-NC 4.0 - see /LICENSE

data_dir=${1:-/.../DTU}
mesh_name=${2:-mesh_2dgs.ply}
output_name=${3:-2dgs}

scans="scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122"

for s in $scans; do
    python scripts/mesh_to_depth.py \
        --mesh_file=${data_dir}/${s}/${mesh_name} \
        --colmap_sparse_folder=${data_dir}/${s}/sparse \
        --output_depth_dir=${data_dir}/${s}/depth_${output_name} \
        --output_masks_dir=${data_dir}/${s}/masks_${output_name}
done
