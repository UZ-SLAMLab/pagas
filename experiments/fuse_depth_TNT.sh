#!/bin/bash
# License: CC BY-NC 4.0 - see /LICENSE

# Default values for tnt_360_scenes
data_folder="/.../TNT"
scans="Barn Caterpillar Ignatius Truck"
relative_path=""
depth_name="depth"
mask_name=""
mesh_name="mesh.ply"
depth_trunc=3.0
voxel_size=0.004
sdf_trunc=4.0
mesh_res=1024
min_mesh_size=50
num_cluster=1

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --data_folder=*) data_folder="${arg#*=}" ;;
        --scans=*) scans="${arg#*=}" ;;
        --relative_path=*) relative_path="${arg#*=}" ;;
        --depth_name=*) depth_name="${arg#*=}" ;;
        --mask_name=*) mask_name="${arg#*=}" ;;
        --mesh_name=*) mesh_name="${arg#*=}" ;;
        --depth_trunc=*) depth_trunc="${arg#*=}" ;;
        --voxel_size=*) voxel_size="${arg#*=}" ;;
        --sdf_trunc=*) sdf_trunc="${arg#*=}" ;;
        --mesh_res=*) mesh_res="${arg#*=}" ;;
        --min_mesh_size=*) min_mesh_size="${arg#*=}" ;;
        --num_cluster=*) num_cluster="${arg#*=}" ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Launch fusion for each scan
for s in $scans; do
    python scripts/fuse_depth.py \
        --data_dir="${data_folder}/${s}/${relative_path}" \
        --depth_name="${depth_name}" \
        --mask_name="${mask_name}" \
        --depth_trunc="${depth_trunc}" \
        --voxel_size="${voxel_size}" \
        --sdf_trunc="${sdf_trunc}" \
        --mesh_res="${mesh_res}" \
        --min_mesh_size="${min_mesh_size}" \
        --num_cluster="${num_cluster}" \
        --mesh_name="${mesh_name}"
done


# Default values for tnt_large_scenes
data_folder="/.../TNT"
scans="Meetingroom Courthouse"
relative_path=""
depth_name="depth"
mask_name=""
mesh_name="mesh.ply"
depth_trunc=4.5
voxel_size=0.006
sdf_trunc=4.0
mesh_res=1024
min_mesh_size=50
num_cluster=1

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --data_folder=*) data_folder="${arg#*=}" ;;
        --scans=*) scans="${arg#*=}" ;;
        --relative_path=*) relative_path="${arg#*=}" ;;
        --depth_name=*) depth_name="${arg#*=}" ;;
        --mask_name=*) mask_name="${arg#*=}" ;;
        --mesh_name=*) mesh_name="${arg#*=}" ;;
        --depth_trunc=*) depth_trunc="${arg#*=}" ;;
        --voxel_size=*) voxel_size="${arg#*=}" ;;
        --sdf_trunc=*) sdf_trunc="${arg#*=}" ;;
        --mesh_res=*) mesh_res="${arg#*=}" ;;
        --min_mesh_size=*) min_mesh_size="${arg#*=}" ;;
        --num_cluster=*) num_cluster="${arg#*=}" ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Launch fusion for each scan
for s in $scans; do
    python scripts/fuse_depth.py \
        --data_dir="${data_folder}/${s}/${relative_path}" \
        --depth_name="${depth_name}" \
        --mask_name="${mask_name}" \
        --depth_trunc="${depth_trunc}" \
        --voxel_size="${voxel_size}" \
        --sdf_trunc="${sdf_trunc}" \
        --mesh_res="${mesh_res}" \
        --min_mesh_size="${min_mesh_size}" \
        --num_cluster="${num_cluster}" \
        --mesh_name="${mesh_name}"
done