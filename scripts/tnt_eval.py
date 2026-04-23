# License: CC BY-NC 4.0 - see /LICENSE
import os
from argparse import ArgumentParser

tnt_360_scenes = ['Barn', 'Caterpillar', 'Ignatius', 'Truck']
tnt_large_scenes = ['Meetingroom', 'Courthouse']

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument('--data_folder', required=True, type=str)
parser.add_argument("--relative_path_to_mesh_360", required=True, help="Relative path to the mesh from the data_folder, e.g., 'results/depth_2dgs_pagas/mesh_360.ply'")
parser.add_argument("--relative_path_to_mesh_large", required=True, help="Relative path to the mesh from the data_folder, e.g., 'results/depth_2dgs_pagas/mesh_large.ply'")
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()


script_dir = os.path.dirname(os.path.abspath(__file__))

for scene in tnt_360_scenes:
    ply_file = os.path.join(args.data_folder, scene, args.relative_path_to_mesh_360)
    string = f"OMP_NUM_THREADS=4 python {script_dir}/eval_tnt/run.py " + \
        f"--dataset-dir {args.data_folder}/{scene} " + \
        f"--traj-path {args.data_folder}/{scene}/{scene}_COLMAP_SfM.log " + \
        f"--ply-path {ply_file}" + \
        f" --out-dir {args.output_dir}" 
    print(string)
    os.system(string)

for scene in tnt_large_scenes:
    ply_file = os.path.join(args.data_folder, scene, args.relative_path_to_mesh_large)
    string = f"OMP_NUM_THREADS=4 python {script_dir}/eval_tnt/run.py " + \
        f"--dataset-dir {args.data_folder}/{scene} " + \
        f"--traj-path {args.data_folder}/{scene}/{scene}_COLMAP_SfM.log " + \
        f"--ply-path {ply_file}" + \
        f" --out-dir {args.output_dir}" 
    print(string)
    os.system(string)