# License: CC BY-NC 4.0 - see /LICENSE
import os
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, help='e.g. "/path/to/TNT/results"')
args = parser.parse_args()

scenes = ["Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck"]


methods = [d for d in os.listdir(args.results_path) if os.path.isdir(os.path.join(args.results_path, d))]
methods.sort()

data = {method: [] for method in methods}

for method in methods:
    for scene in scenes:
        file_path = os.path.join(args.results_path, method, f"{scene}.prf_tau_plotstr.txt")
        value = float('nan')
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r") as f:
                    lines = [ln.strip() for ln in f.readlines()]
                if len(lines) >= 3:
                    value = float(lines[2])
            except Exception:
                value = float('nan')
        data[method].append(value)

df = pd.DataFrame(data, index=scenes)
df.index.name = "Scene"

# Compute mean with all decimal precision
mean_values = df.mean(skipna=True)
mean_row = pd.DataFrame([mean_values], index=["Mean"])

# Combine into one DataFrame
df_with_mean = pd.concat([df, mean_row])

# Format for display
def format_value(val):
    if pd.isna(val):
        return "nan"
    return f"{val:.4f}"

df_pretty = df_with_mean.applymap(format_value)

print("Per scene values and mean:")
print(df_pretty)