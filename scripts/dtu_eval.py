# License: CC BY-NC 4.0 - see /LICENSE
import os
from argparse import ArgumentParser


dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--data_folder", required=True)
parser.add_argument("--relative_path_to_mesh", required=True, help="Relative path to the mesh from the data_folder, e.g., 'results/depth_2dgs_pagas/mesh.ply'")
parser.add_argument("--output_dir", required=True)
parser.add_argument('--DTU_groundtruth', required=True, type=str)
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))

for scene in dtu_scenes:
    scan_id = scene[4:]
    string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
        f"--input_mesh {os.path.join(args.data_folder, scene, args.relative_path_to_mesh)} " + \
        f"--scan_id {scan_id} --output_dir {os.path.join(args.output_dir, scan_id)} " + \
        f"--DTU {args.DTU_groundtruth}" + \
        f" --data_folder {args.data_folder}"

    os.system(string)
