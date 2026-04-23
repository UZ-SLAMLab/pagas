#!/usr/bin/env bash
# License: CC BY-NC 4.0 - see /LICENSE

# Parse arguments
for arg in "$@"; do
    case $arg in
        --path_DTU=*) path_DTU="${arg#*=}" ;;
        --path_DTU_gt=*) path_DTU_gt="${arg#*=}" ;;
        --path_TNT=*) path_TNT="${arg#*=}" ;;
        --conda_env_tnt=*) conda_env_tnt="${arg#*=}" ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

conda_activate() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: conda not found in PATH."
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$1"
}

conda_activate "pagas"

# Extract initial depth:
./experiments/mesh_to_depth_DTU.sh $path_DTU mesh_mvsa.ply mvsa
./experiments/mesh_to_depth_TNT.sh $path_TNT mesh_mvsa.ply mesh_mvsa.ply mvsa

./experiments/mesh_to_depth_DTU.sh $path_DTU mesh_2dgs.ply 2dgs
./experiments/mesh_to_depth_TNT.sh $path_TNT mesh_2dgs.ply mesh_2dgs.ply 2dgs

./experiments/mesh_to_depth_DTU.sh $path_DTU mesh_pgsr.ply pgsr
./experiments/mesh_to_depth_TNT.sh $path_TNT mesh_pgsr.ply mesh_pgsr.ply pgsr

# Refine their depth with PAGaS:
./experiments/run_pagas_DTU.sh $path_DTU depth_mvsa masks_mvsa
./experiments/run_pagas_TNT.sh $path_TNT depth_mvsa masks_mvsa
./experiments/run_pagas_TNT_exposure.sh $path_TNT depth_mvsa masks_mvsa

./experiments/run_pagas_DTU.sh $path_DTU depth_2dgs masks_2dgs
./experiments/run_pagas_TNT.sh $path_TNT depth_2dgs masks_2dgs
./experiments/run_pagas_TNT_exposure.sh $path_TNT depth_2dgs masks_2dgs

./experiments/run_pagas_DTU.sh $path_DTU depth_pgsr masks_pgsr
./experiments/run_pagas_TNT.sh $path_TNT depth_pgsr masks_pgsr
./experiments/run_pagas_TNT_exposure.sh $path_TNT depth_pgsr masks_pgsr

# Stereo depth consistency filtering only in TnT with exposure compensation:
./experiments/consistent_depth_TNT.sh --data_folder=$path_TNT --relative_path="results/depth_mvsa_pagas_exposure"
./experiments/consistent_depth_TNT.sh --data_folder=$path_TNT --relative_path="results/depth_2dgs_pagas_exposure"
./experiments/consistent_depth_TNT.sh --data_folder=$path_TNT --relative_path="results/depth_pgsr_pagas_exposure"

# Fuse depth into mesh:
./experiments/fuse_depth_DTU.sh --data_folder=$path_DTU --relative_path="results/depth_mvsa_pagas"
./experiments/fuse_depth_TNT.sh --data_folder=$path_TNT --relative_path="results/depth_mvsa_pagas" --num_cluster=50
./experiments/fuse_depth_TNT.sh --data_folder=$path_TNT --relative_path="results/depth_mvsa_pagas_exposure" --depth_name="depth_consistent" --num_cluster=50

./experiments/fuse_depth_DTU.sh --data_folder=$path_DTU --relative_path="results/depth_2dgs_pagas"
./experiments/fuse_depth_TNT.sh --data_folder=$path_TNT --relative_path="results/depth_2dgs_pagas"
./experiments/fuse_depth_TNT.sh --data_folder=$path_TNT --relative_path="results/depth_2dgs_pagas_exposure" --depth_name="depth_consistent"

./experiments/fuse_depth_DTU.sh --data_folder=$path_DTU --relative_path="results/depth_pgsr_pagas"
./experiments/fuse_depth_TNT.sh --data_folder=$path_TNT --relative_path="results/depth_pgsr_pagas"
./experiments/fuse_depth_TNT.sh --data_folder=$path_TNT --relative_path="results/depth_pgsr_pagas_exposure" --depth_name="depth_consistent"

# Evaluation:
python scripts/dtu_eval.py --data_folder $path_DTU --relative_path_to_mesh results/depth_mvsa_pagas/mesh_trunc3_vox0.001_sdf5_post.ply --DTU_groundtruth $path_DTU_gt --output_dir $path_DTU/results/mvsa_pagas
python scripts/dtu_eval.py --data_folder $path_DTU --relative_path_to_mesh results/depth_2dgs_pagas/mesh_trunc3_vox0.001_sdf5_post.ply --DTU_groundtruth $path_DTU_gt --output_dir $path_DTU/results/2dgs_pagas
python scripts/dtu_eval.py --data_folder $path_DTU --relative_path_to_mesh results/depth_pgsr_pagas/mesh_trunc3_vox0.001_sdf5_post.ply --DTU_groundtruth $path_DTU_gt --output_dir $path_DTU/results/pgsr_pagas

conda activate $conda_env_tnt
python scripts/tnt_eval.py --data_folder $path_TNT --relative_path_to_mesh_360 results/depth_mvsa_pagas/mesh_trunc3_vox0.004_sdf4_post.ply --relative_path_to_mesh_large results/depth_mvsa_pagas/mesh_trunc4.5_vox0.006_sdf4_post.ply --output_dir $path_TNT/results/mvsa_pagas
python scripts/tnt_eval.py --data_folder $path_TNT --relative_path_to_mesh_360 results/depth_2dgs_pagas/mesh_trunc3_vox0.004_sdf4_post.ply --relative_path_to_mesh_large results/depth_2dgs_pagas/mesh_trunc4.5_vox0.006_sdf4_post.ply --output_dir $path_TNT/results/2dgs_pagas
python scripts/tnt_eval.py --data_folder $path_TNT --relative_path_to_mesh_360 results/depth_pgsr_pagas/mesh_trunc3_vox0.004_sdf4_post.ply --relative_path_to_mesh_large results/depth_pgsr_pagas/mesh_trunc4.5_vox0.006_sdf4_post.ply --output_dir $path_TNT/results/pgsr_pagas

python scripts/tnt_eval.py --data_folder $path_TNT --relative_path_to_mesh_360 results/depth_mvsa_pagas_exposure/mesh_trunc3_vox0.004_sdf4_post.ply --relative_path_to_mesh_large results/depth_mvsa_pagas_exposure/mesh_trunc4.5_vox0.006_sdf4_post.ply --output_dir $path_TNT/results/mvsa_pagas_exposure
python scripts/tnt_eval.py --data_folder $path_TNT --relative_path_to_mesh_360 results/depth_2dgs_pagas_exposure/mesh_trunc3_vox0.004_sdf4_post.ply --relative_path_to_mesh_large results/depth_2dgs_pagas_exposure/mesh_trunc4.5_vox0.006_sdf4_post.ply --output_dir $path_TNT/results/2dgs_pagas_exposure
python scripts/tnt_eval.py --data_folder $path_TNT --relative_path_to_mesh_360 results/depth_pgsr_pagas_exposure/mesh_trunc3_vox0.004_sdf4_post.ply --relative_path_to_mesh_large results/depth_pgsr_pagas_exposure/mesh_trunc4.5_vox0.006_sdf4_post.ply --output_dir $path_TNT/results/pgsr_pagas_exposure
conda_activate "pagas"

# Read results
python scripts/eval_dtu/read_results_DTU.py --results_path $path_DTU/results
python scripts/eval_tnt/read_results_TNT.py --results_path $path_TNT/results
