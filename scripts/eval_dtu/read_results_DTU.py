# License: CC BY-NC 4.0 - see /LICENSE
import os
import json
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, required=True, help='e.g. "/path/to/DTU/results"')
args = parser.parse_args()

methods = ["mvsa_pagas", "2dgs_pagas", "pgsr_pagas"]
scans = ["24", "37", "40", "55", "63", "65", "69", "83", "97", "105", "106", "110", "114", "118", "122"]


data = {method: [] for method in methods}

for method in methods:
    for scan in scans:
        results_file = os.path.join(args.results_path, method, scan, "results.json")
        try:
            with open(results_file, "r") as f:
                result = json.load(f)
                overall = result.get("overall", None)
        except Exception:
            overall = None
        data[method].append(overall)

df = pd.DataFrame(data, index=scans)
df.index.name = "Scan"
pd.options.display.float_format = "{:.4f}".format

print("Per-scene values:")
print(df)

print("\nMean for each method:")
print(df.mean(skipna=True).round(2))
