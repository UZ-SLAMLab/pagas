#!/bin/bash
# License: CC BY-NC 4.0 - see /LICENSE

data_dir=${1:-/.../DTU}
mesh_name_360=${2:-mesh_2dgs.ply}
mesh_name_large=${3:-mesh_2dgs.ply}
output_name=${4:-2dgs}

scans="Barn Caterpillar Ignatius Truck"

for s in $scans; do
    python scripts/mesh_to_depth.py \
        --mesh_file=${data_dir}/${s}/${mesh_name_360} \
        --colmap_sparse_folder=${data_dir}/${s}/sparse \
        --output_depth_dir=${data_dir}/${s}/depth_${output_name} \
        --output_masks_dir=${data_dir}/${s}/masks_${output_name}
done


scans="Meetingroom Courthouse"

for s in $scans; do
    python scripts/mesh_to_depth.py \
        --mesh_file=${data_dir}/${s}/${mesh_name_large} \
        --colmap_sparse_folder=${data_dir}/${s}/sparse \
        --output_depth_dir=${data_dir}/${s}/depth_${output_name} \
        --output_masks_dir=${data_dir}/${s}/masks_${output_name}
done