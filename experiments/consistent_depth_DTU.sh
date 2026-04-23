#!/bin/bash
# License: CC BY-NC 4.0 - see /LICENSE

# Default values
data_folder=".../DTU"
relative_path="results/pagas"
scans="scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122"

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
