#!/bin/bash
# License: CC BY-NC 4.0 - see /LICENSE

# Default values
data_folder=".../TNT"
relative_path="results/pagas"
scans="Barn Caterpillar Ignatius Truck Meetingroom Courthouse"

# Parse named arguments
for arg in "$@"; do
    case $arg in
        --data_folder=*) data_folder="${arg#*=}" ;;
        --relative_path=*) relative_path="${arg#*=}" ;;
    esac
done

# Launch fusion for each scan
for s in $scans; do
    python scripts/consistent_depth.py \
        --data_folder="${data_folder}/${s}/${relative_path}"
done
