#!/usr/bin/env bash
# License: CC BY-NC 4.0 - see /LICENSE
set -euo pipefail

# Default values
data_folder="/.../scan24"
depth_name="depth"
mask_name="masks"
image_name="images"
mesh_name="mesh.ply"
depth_trunc=-1.0
voxel_size=-1.0
sdf_trunc=5.0
mesh_res=2048
min_mesh_size=50
num_cluster=1
erode_borders=false

# Simple key=value parsing
for arg in "$@"; do
  case $arg in
    --data_folder=*) data_folder="${arg#*=}" ;;
    --depth_name=*) depth_name="${arg#*=}" ;;
    --mask_name=*) mask_name="${arg#*=}" ;;
    --image_name=*) image_name="${arg#*=}" ;;
    --mesh_name=*) mesh_name="${arg#*=}" ;;
    --depth_trunc=*) depth_trunc="${arg#*=}" ;;
    --voxel_size=*) voxel_size="${arg#*=}" ;;
    --sdf_trunc=*) sdf_trunc="${arg#*=}" ;;
    --mesh_res=*) mesh_res="${arg#*=}" ;;
    --min_mesh_size=*) min_mesh_size="${arg#*=}" ;;
    --num_cluster=*) num_cluster="${arg#*=}" ;;
    --erode_borders) erode_borders=true ;;
    *) echo "   Unknown argument: $arg"; exit 1 ;;
  esac
done

data_dir="${data_folder}"
img_dir="${data_folder}/${image_name}"
depth_dir="${data_dir}/${depth_name}"
mask_dir="${data_dir}/${mask_name}"

# Basic validations
[[ -d "$data_dir" ]] || { echo "   data_dir does not exist: $data_dir"; exit 2; }
[[ -d "$img_dir" ]] || { echo "   image dir does not exist: $img_dir"; exit 2; }
[[ -d "$depth_dir" ]] || { echo "   depth dir does not exist: $depth_dir"; exit 2; }

n_img=$(find "$img_dir" -maxdepth 1 -type f | wc -l || echo 0)
n_depth=$(find "$depth_dir" -maxdepth 1 -type f -name "*.npz" | wc -l || echo 0)
echo "   Images: $n_img. Depths: $n_depth. Data dir: $data_dir"

if [[ -n "${mask_name}" && "${mask_name}" != "" ]]; then
  if [[ ! -d "$mask_dir" ]]; then
    echo "   Warning. mask_name='${mask_name}' but dir does not exist: $mask_dir. It will be omitted."
    mask_name=""
  fi
fi

# Build optional args only if applicable
extra_args=()
[[ -n "$image_name" ]] && extra_args+=("--image_name=${image_name}")
[[ -n "$depth_name" ]] && extra_args+=("--depth_name=${depth_name}")
if [[ -n "$mask_name" ]]; then
  extra_args+=("--mask_name=${mask_name}")
fi

# Run with faulthandler enabled on CPU
export PYTHONFAULTHANDLER=1
export OPEN3D_CPU_RELEASE=1

py_args=(
  "scripts/fuse_depth.py"
  "--data_dir=${data_dir}"
  "${extra_args[@]}"
  "--depth_trunc=${depth_trunc}"
  "--voxel_size=${voxel_size}"
  "--sdf_trunc=${sdf_trunc}"
  "--mesh_res=${mesh_res}"
  "--min_mesh_size=${min_mesh_size}"
  "--num_cluster=${num_cluster}"
  "--mesh_name=${mesh_name}"
)

if [[ "$erode_borders" == "true" ]]; then
  py_args+=( "--erode_borders" )
fi

python "${py_args[@]}"
