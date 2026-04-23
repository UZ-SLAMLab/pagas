#!/usr/bin/env bash
# License: CC BY-NC 4.0 - see /LICENSE

set -euo pipefail


# ------------------ Normalize leading dashes in arguments ------------------
is_dashlike() {
  local c="$1"
  [[ "$c" == "-" || "$c" == $'\u2010' || "$c" == $'\u2011' || "$c" == $'\u2012' || "$c" == $'\u2013' || "$c" == $'\u2014' || "$c" == $'\u2212' ]]
}

normalize_leading_dashes() {
  local a="$1"
  local i=0 n=${#a}
  # count how many leading dash-like characters there are
  while (( i < n )); do
    local ch="${a:i:1}"
    is_dashlike "$ch" || break
    (( i++ ))
  done
  # if none, return as-is
  if (( i == 0 )); then
    printf '%s' "$a"
    return
  fi
  # collapse 1 or more dash-like chars -> '-' (for 1) or '--' (for 2+)
  local prefix="-" 
  if (( i >= 2 )); then
    prefix="--"
  fi
  printf '%s%s' "$prefix" "${a:i}"
}

# Rebuild $@ with normalized leading dashes
__norm_args=()
for __arg in "$@"; do
  __norm_args+=("$(normalize_leading_dashes "$__arg")")
done
set -- "${__norm_args[@]}"
unset __arg __norm_args

# ------------------ Parse arguments ------------------
DATA_FOLDER=""
PATH_MVS_METHOD=""
GET_MASKS=false
MESH_RES=4000
NO_UNDIST=false
NO_SHARED_INTRINSICS=false
VOXEL_SIZE_OVERRIDE=""
DEPTH_TRUNC_OVERRIDE=""
EXPOSURE_ARG=""
SAVE_EXTRA_ARG=""
CONSISTENT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_folder)
      [[ $# -ge 2 ]] || { echo "Error: --data_folder requires a value"; exit 1; }
      DATA_FOLDER="$2"; shift 2 ;;
    --data_folder=*)
      DATA_FOLDER="${1#*=}"; shift ;;

    --path_mvs_method)
      [[ $# -ge 2 ]] || { echo "Error: --path_mvs_method requires a value"; exit 1; }
      PATH_MVS_METHOD="$2"; shift 2 ;;
    --path_mvs_method=*)
      PATH_MVS_METHOD="${1#*=}"; shift ;;

    --get_masks|--get_masks=*)
      GET_MASKS=true; shift ;;

    --mesh_res)
      [[ $# -ge 2 ]] || { echo "Error: --mesh_res requires a value"; exit 1; }
      MESH_RES="$2"; shift 2 ;;
    --mesh_res=*)
      MESH_RES="${1#*=}"; shift ;;

    --no_undist|--no_undist=*)
      NO_UNDIST=true; shift ;;

    --no_shared_intrinsics|--no_shared_intrinsics=*)
      NO_SHARED_INTRINSICS=true; shift ;;

    --voxel_size)
      [[ $# -ge 2 ]] || { echo "Error: --voxel_size requires a value"; exit 1; }
      VOXEL_SIZE_OVERRIDE="$2"; shift 2 ;;
    --voxel_size=*)
      VOXEL_SIZE_OVERRIDE="${1#*=}"; shift ;;

    --depth_trunc)
      [[ $# -ge 2 ]] || { echo "Error: --depth_trunc requires a value"; exit 1; }
      DEPTH_TRUNC_OVERRIDE="$2"; shift 2 ;;
    --depth_trunc=*)
      DEPTH_TRUNC_OVERRIDE="${1#*=}"; shift ;;

    --exposure|--exposure=*)
      EXPOSURE_ARG="--exposure"; shift ;;

    --save_extra|--save_extra=*)
      SAVE_EXTRA_ARG="--save_extra"; shift ;;

    --consistent|--consistent=*)
      CONSISTENT=true; shift ;;

    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --data_folder PATH [--path_mvs_method PATH_METHOD] [--get_masks] [--mesh_res INT] [--voxel_size FLOAT] [--depth_trunc FLOAT] [--no_undist] [--no_shared_intrinsics] [--exposure] [--consistent] [--save_extra]"
      exit 1 ;;
  esac
done

# ------------------ Prefer existing depth_init over --path_mvs_method ------------------
DEPTH_INIT_DIR=""
MVS_METHOD="${MVS_METHOD:-}"
PATH_MVS_METHOD="${PATH_MVS_METHOD:-}"
MESH_INIT=""
DEPTH_CUSTOM_DIR=""

if [[ -d "$DATA_FOLDER/undist/depth_init" ]]; then
    DEPTH_INIT_DIR="$DATA_FOLDER/undist/depth_init"
elif [[ -d "$DATA_FOLDER/depth_init" ]]; then
    DEPTH_INIT_DIR="$DATA_FOLDER/depth_init"
fi

if [[ -d "$DATA_FOLDER/undist/depth" ]]; then
    DEPTH_CUSTOM_DIR="$DATA_FOLDER/undist/depth"
elif [[ -d "$DATA_FOLDER/depth" ]]; then
    DEPTH_CUSTOM_DIR="$DATA_FOLDER/depth"
fi

if [[ -n "${DEPTH_INIT_DIR:-}" ]]; then
    echo -e "\033[1;34m[SKIP] Found initial depths at '$DEPTH_INIT_DIR'\033[0m"
    echo -e "\033[1;34m[SKIP] PAGaS will refine the depths there, ignoring --path_mvs_method if provided\033[0m"
    PATH_MVS_METHOD="none"; MVS_METHOD="none"

elif [[ -f "$DATA_FOLDER/undist/mesh_init.ply" ]]; then
    MESH_INIT="$DATA_FOLDER/undist/mesh_init.ply"
    echo -e "\033[1;34m[SKIP] Found existing mesh_init.ply at '$MESH_INIT'. Skipping initial MVS. Extracting depths from it.\033[0m"
    PATH_MVS_METHOD="none"; MVS_METHOD="none"

elif [[ -f "$DATA_FOLDER/mesh_init.ply" ]]; then
    MESH_INIT="$DATA_FOLDER/mesh_init.ply"
    echo -e "\033[1;34m[SKIP] Found existing mesh_init.ply at '$MESH_INIT'. Skipping initial MVS. Extracting depths from it.\033[0m"
    PATH_MVS_METHOD="none"; MVS_METHOD="none"

elif [[ -n "${DEPTH_CUSTOM_DIR:-}" ]]; then
    echo -e "\033[1;34m[SKIP] Found custom unconsistent depths at '$DEPTH_CUSTOM_DIR'\033[0m"
    echo -e "\033[1;34m[SKIP] They will be fused into an initial mesh and the initial depths extracted from that, ignoring --path_mvs_method if provided\033[0m"
    PATH_MVS_METHOD="none"; MVS_METHOD="none"

else
    # ------------------ If --path_mvs_method is given, infer MVS_METHOD ------------------
    if [[ -n "$PATH_MVS_METHOD" ]]; then
        _full_lc="$(printf '%s' "$PATH_MVS_METHOD" | tr '[:upper:]' '[:lower:]')"
        _base_lc="$(basename "$PATH_MVS_METHOD" | tr '[:upper:]' '[:lower:]')"

        if [[ "$_base_lc" =~ ^(2d-gaussian-splatting|2d_gaussian_splatting|2dgs)$ ]] \
           || [[ "$_full_lc" == *"2d-gaussian-splatting"* ]] \
           || [[ "$_full_lc" == *"2d_gaussian_splatting"* ]]; then
            MVS_METHOD="2dgs"
        elif [[ "$_base_lc" =~ ^(pgsr|pixel-aligned-gaussian-splatting|pixel_aligned_gaussian_splatting)$ ]] \
             || [[ "$_full_lc" == *"pgsr"* ]]; then
            MVS_METHOD="pgsr"
        elif [[ "$_base_lc" =~ ^(mvsanywhere|mvs-anywhere|mvsa)$ ]] \
             || [[ "$_full_lc" == *"mvsanywhere"* ]] \
             || [[ "$_full_lc" == *"mvs-anywhere"* ]]; then
            MVS_METHOD="mvsa"
        fi

        if [[ "$MVS_METHOD" != "pgsr" && "$MVS_METHOD" != "2dgs" && "$MVS_METHOD" != "mvsa" ]]; then
            echo "Error: could not infer MVS_METHOD from --path_mvs_method ('$PATH_MVS_METHOD')."
            echo "Expected the path to include one of: 2d-gaussian-splatting, PGSR, MVSAnywhere."
            exit 1
        fi
    else
        echo "Error: No --path_mvs_method provided and no mesh or initial depths found."
        echo "Either place your custom initial mesh in '$DATA_FOLDER/mesh_init.ply',"
        echo "  your depth extracted from your custom initial mesh in '$DATA_FOLDER/depth_init',"
        echo "  your custom unconsistent depth in '$DATA_FOLDER/depth',"
        echo "  or provide --path_mvs_method to estimate the initial mesh."
        exit 1
    fi
fi


# ------------------ Validate arguments ------------------
if [[ -z "${DATA_FOLDER:-}" ]]; then
    echo "Error: --data_folder is required."
    exit 1
fi

if ! [[ "$MESH_RES" =~ ^[0-9]+$ ]]; then
    echo "Error: --mesh_res must be an integer."
    exit 1
fi

if [[ ! -d "$DATA_FOLDER" ]]; then
    echo "Error: data folder not found: $DATA_FOLDER"
    exit 1
fi

# Detect if the user already provided masks at the root
HAD_INPUT_MASKS=false
if [[ -d "$DATA_FOLDER/masks" ]]; then
    HAD_INPUT_MASKS=true
fi

# Resolve repo root so we can call scripts/run_colmap.sh reliably
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"


# ------------------ Undistortion policy (no_undist or existing data) ------------------
HAS_LOCAL_MESH_INIT=false
HAS_LOCAL_DEPTH_INIT=false
HAS_LOCAL_DEPTH=false

[[ -f "$DATA_FOLDER/mesh_init.ply" ]] && HAS_LOCAL_MESH_INIT=true
[[ -d "$DATA_FOLDER/depth_init" ]] && HAS_LOCAL_DEPTH_INIT=true
[[ -d "$DATA_FOLDER/depth" ]] && HAS_LOCAL_DEPTH=true

FORCE_NO_UNDIST=false
if $NO_UNDIST || $HAS_LOCAL_MESH_INIT || $HAS_LOCAL_DEPTH_INIT || $HAS_LOCAL_DEPTH; then
    FORCE_NO_UNDIST=true
fi

if $FORCE_NO_UNDIST; then
    echo -e "\033[1;34m[UNDIST] Using '$DATA_FOLDER' as effective dataset without image undistortion (reason: --no_undist or existing mesh/depth in root)\033[0m"
fi


# ------------------ Helpers ------------------
require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command not found in PATH: $cmd"
        exit 1
    fi
}

format5() {
    local x="$1"

    # Numeric check
    if [[ "$x" =~ ^-?[0-9]*\.?[0-9]+$ ]]; then
        local s sign int frac frac5

        # Fixed decimal with enough precision
        s=$(LC_NUMERIC=C printf '%.10f' "$x")

        # Exact zero
        if [[ "$s" == "0.0000000000" || "$s" == "-0.0000000000" ]]; then
            printf '0'
            return
        fi

        # Separate sign
        sign=""
        if [[ "$s" == -* ]]; then
            sign="-"
            s="${s#-}"
        fi

        # Split integer and fractional parts
        int="${s%%.*}"
        frac="${s#*.}"

        # First 5 decimal digits, pad if shorter
        frac5="${frac:0:5}"
        while ((${#frac5} < 5)); do
            frac5="${frac5}0"
        done

        if [[ "$int" != "0" || "$frac5" =~ [1-9] ]]; then
            # Decimal representation, cut at 5 decimals, no rounding
            while [[ -n "$frac5" && "${frac5: -1}" == 0 ]]; do
                frac5="${frac5%0}"
            done
            if [[ -n "$frac5" ]]; then
                printf '%s%s.%s' "$sign" "$int" "$frac5"
            else
                printf '%s%s' "$sign" "$int"
            fi
        else
            # Scientific notation for very small numbers like 0.000001
            local sci ssign mant exp mant_int mant_frac digits digits5

            sci=$(LC_NUMERIC=C printf '%.10e' "$x")

            ssign=""
            if [[ "$sci" == -* ]]; then
                ssign="-"
                sci="${sci#-}"
            fi

            mant="${sci%e*}"
            exp="e${sci#*e}"

            mant_int="${mant%%.*}"
            mant_frac="${mant#*.}"
            digits="${mant_int}${mant_frac}"

            digits5="${digits:0:5}"
            while ((${#digits5} < 5)); do
                digits5="${digits5}0"
            done

            # Trim trailing zeros but keep at least one digit
            while [[ ${#digits5} -gt 1 && "${digits5: -1}" == 0 ]]; do
                digits5="${digits5%0}"
            done

            if [[ ${#digits5} -gt 1 ]]; then
                printf '%s%s.%s%s' "$ssign" "${digits5:0:1}" "${digits5:1}" "$exp"
            else
                printf '%s%s%s' "$ssign" "$digits5" "$exp"
            fi
        fi
    else
        # Non numeric, passthrough
        printf '%s' "$x"
    fi
}

has_any_images() {
    # return 0 if images folder exists and has at least one png or jpg
    local img_dir="$1/images"
    if [[ -d "$img_dir" ]]; then
        if find "$img_dir" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print -quit | grep -q .; then
            return 0
        fi
    fi
    return 1
}

find_videos() {
    # prints video paths, one per line
    find "$DATA_FOLDER" -maxdepth 1 -type f \( \
        -iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" -o -iname "*.mkv" -o \
        -iname "*.m4v" -o -iname "*.webm" -o -iname "*.mpg" -o -iname "*.mpeg" \
    \)
}

ensure_images_ready() {
    # 1) if images exist and non empty, do nothing
    # 2) else look for exactly one video and extract frames into images
    if has_any_images "$DATA_FOLDER"; then
        echo -e "\033[1;34m[Images] Found images in $DATA_FOLDER/images\033[0m"
        return 0
    fi

    echo -e "\033[1;34m[Images] images folder missing or empty. Looking for a single video in $DATA_FOLDER\033[0m"
    mapfile -t VIDEO_FILES < <(find_videos)

    if (( ${#VIDEO_FILES[@]} == 0 )); then
        echo "Error: neither '$DATA_FOLDER/images' with images nor a video file found."
        echo "Place an 'images' folder with frames or a single video in $DATA_FOLDER."
        exit 1
    fi

    if (( ${#VIDEO_FILES[@]} > 1 )); then
        echo "Error: more than one video found in $DATA_FOLDER. Only one video is allowed for frame extraction."
        printf 'Found videos:\n'
        printf '  %s\n' "${VIDEO_FILES[@]}"
        exit 1
    fi

    local video="${VIDEO_FILES[0]}"
    echo -e "\033[1;34m[Images] Extracting frames from: $(basename "$video")\033[0m"
    require_cmd ffmpeg
    mkdir -p "$DATA_FOLDER/images"
    ffmpeg -y -i "$video" -vsync 0 -start_number 0 "$DATA_FOLDER/images/%06d.png"
    # sanity check
    if ! has_any_images "$DATA_FOLDER"; then
        echo "Error: frame extraction failed, images folder still empty."
        exit 1
    fi
    echo -e "\033[1;34m[Images] Extraction complete\033[0m"
}

extract_masks() {
    echo -e "\033[1;34m[Masks] Extracting binary masks using rembg then thresholding\033[0m"
    require_cmd rembg
    require_cmd mogrify
    pushd "$DATA_FOLDER" >/dev/null
    # rembg will create the 'masks' folder and mirror filenames
    rembg p -m birefnet-massive -om images masks
    if ! [[ -d "masks" ]]; then
        echo "Error: rembg did not create the masks folder."
        popd >/dev/null
        exit 1
    fi
    # convert to binary masks
    mogrify -colorspace Gray -threshold 50% masks/*
    popd >/dev/null
    echo -e "\033[1;34m[Masks] Masks ready at $DATA_FOLDER/masks\033[0m"
}

run_colmap_pipeline() {
    local dataset_dir="$1"
    local do_undist="$2"  # "true" or "false"

    local intrinsics_msg
    if [[ "$NO_SHARED_INTRINSICS" == "true" ]]; then
        intrinsics_msg="without shared intrinsics"
    else
        intrinsics_msg="with shared intrinsics"
    fi

    local masks_src_dir="$dataset_dir/masks"
    local have_masks=false
    local tmp_masks_dir=""
    local will_undistort_masks=false

    # We only care about undistorting masks when:
    #   1) undistortion is requested
    #   2) a masks folder exists at dataset_dir/masks
    if [[ "$do_undist" == "true" && -d "$masks_src_dir" ]]; then
        have_masks=true
        will_undistort_masks=true
        # We temporarily move masks away so run_colmap.sh never touches them.
        # Later we will undistort them ourselves with colmap image_undistorter.
        tmp_masks_dir="${dataset_dir}/.masks_for_undist"
        rm -rf "$tmp_masks_dir"
        mv "$masks_src_dir" "$tmp_masks_dir"
        echo -e "\033[1;34m[Masks] Using detected masks at $dataset_dir/masks.\033[0m"
    fi

    if [[ "$do_undist" == "true" ]]; then
        echo -e "\033[1;34m[COLMAP] Running COLMAP $intrinsics_msg and image undistortion\033[0m"
    else
        echo -e "\033[1;34m[COLMAP] Running COLMAP $intrinsics_msg without image undistortion\033[0m"
    fi

    if [[ ! -x "$REPO_ROOT/scripts/run_colmap.sh" ]]; then
        echo "Error: cannot find executable $REPO_ROOT/scripts/run_colmap.sh"
        # Restore masks if we moved them
        if [[ "$will_undistort_masks" == "true" && -d "$tmp_masks_dir" ]]; then
            mv "$tmp_masks_dir" "$masks_src_dir"
        fi
        exit 1
    fi

    pushd "$REPO_ROOT" >/dev/null

    if [[ "$do_undist" == "true" ]]; then
        if [[ "$NO_SHARED_INTRINSICS" == "true" ]]; then
            bash "scripts/run_colmap.sh" "$dataset_dir" --undistort_images
        else
            bash "scripts/run_colmap.sh" "$dataset_dir" --shared_intrinsics --undistort_images
        fi
    else
        if [[ "$NO_SHARED_INTRINSICS" == "true" ]]; then
            bash "scripts/run_colmap.sh" "$dataset_dir"
        else
            bash "scripts/run_colmap.sh" "$dataset_dir" --shared_intrinsics
        fi
    fi

    popd >/dev/null

    # Mirror sparse/0 files into sparse/ as symlinks, for consistency with undist layout
    if [[ -d "$dataset_dir/sparse/0" ]]; then
        mkdir -p "$dataset_dir/sparse"
        for f in cameras.bin images.bin points3D.bin cameras.txt images.txt points3D.txt; do
            if [[ -f "$dataset_dir/sparse/0/$f" && ! -e "$dataset_dir/sparse/$f" ]]; then
                # create relative symlink pointing to sparse/0/<file>
                ln -s "0/$f" "$dataset_dir/sparse/$f"
            fi
        done
    fi

    if [[ "$do_undist" == "true" ]]; then
        if [[ ! -d "$dataset_dir/undist" ]]; then
            echo "Error: expected undist folder not found at $dataset_dir/undist after COLMAP."
            # Restore masks if we moved them
            if [[ "$will_undistort_masks" == "true" && -d "$tmp_masks_dir" ]]; then
                mv "$tmp_masks_dir" "$masks_src_dir"
            fi
            exit 1
        fi
        echo -e "\033[1;34m[COLMAP] Undistorted dataset ready at $dataset_dir/undist\033[0m"

        # Now undistort the masks with the same COLMAP model if we had masks
        if [[ "$will_undistort_masks" == "true" ]]; then
            require_cmd colmap

            # Find which sparse model COLMAP produced
            local model_dir=""
            if [[ -d "$dataset_dir/sparse/0" ]]; then
                model_dir="$dataset_dir/sparse/0"
            elif compgen -G "$dataset_dir/sparse/*.bin" >/dev/null || compgen -G "$dataset_dir/sparse/*.txt" >/dev/null; then
                model_dir="$dataset_dir/sparse"
            else
                echo -e "\033[1;34m[Masks] Warning: could not find a sparse model under $dataset_dir/sparse to undistort masks. Restoring masks without undistortion.\033[0m"
                mv "$tmp_masks_dir" "$masks_src_dir"
                return 0
            fi

            local masks_undist_root="$dataset_dir/undist/.masks_undist_tmp"
            rm -rf "$masks_undist_root"
            mkdir -p "$masks_undist_root"

            echo -e "\033[1;34m[Masks] Undistorting masks with COLMAP image_undistorter using model at $model_dir\033[0m"
            if ! colmap image_undistorter \
                    --image_path "$tmp_masks_dir" \
                    --input_path "$model_dir" \
                    --output_path "$masks_undist_root" >/dev/null 2>&1; then
                echo -e "\033[1;34m[Masks] Warning: COLMAP mask undistortion failed. Restoring masks without undistortion.\033[0m"
                rm -rf "$masks_undist_root"
                mv "$tmp_masks_dir" "$masks_src_dir"
                return 0
            fi

            mkdir -p "$dataset_dir/undist/masks"
            # colmap image_undistorter writes into <output_path>/images by default
            if [[ -d "$masks_undist_root/images" ]]; then
                cp "$masks_undist_root/images"/* "$dataset_dir/undist/masks/" || true
            else
                # Fallback if COLMAP layout changes
                cp "$masks_undist_root"/* "$dataset_dir/undist/masks/" || true
            fi

            rm -rf "$masks_undist_root"
            mv "$tmp_masks_dir" "$masks_src_dir"

            echo -e "\033[1;34m[Masks] Undistorted masks ready at $dataset_dir/undist/masks\033[0m"
        fi
    else
        echo -e "\033[1;34m[COLMAP] COLMAP run completed on $dataset_dir without undistortion\033[0m"

        # No undistortion, just restore masks if we moved them
        if [[ "$will_undistort_masks" == "true" && -d "$tmp_masks_dir" ]]; then
            mv "$tmp_masks_dir" "$masks_src_dir"
            echo -e "\033[1;34m[Masks] Restored masks at $dataset_dir/masks\033[0m"
        fi
    fi
}


# ------------------ Summary ------------------
echo ""
echo -e "\033[1;38;5;223m======================================="
echo -e "Running automatic pipeline"
echo -e "======================================="
echo -e "Data folder           : $DATA_FOLDER"
echo -e "Using masks           : $([[ "$GET_MASKS" == true || "$HAD_INPUT_MASKS" == true ]] && echo true || echo false)"
echo -e "MVS baseline          : $MVS_METHOD"
echo -e "Mesh resolution       : $MESH_RES"
echo -e "Image undistortion    : $([[ "$FORCE_NO_UNDIST" == "true" ]] && echo "false" || echo "true")"
echo -e "Shared intrinsics     : $([[ "$NO_SHARED_INTRINSICS" == "true" ]] && echo "false" || echo "true")"
echo -e "Exposure compensation : $([[ -n "$EXPOSURE_ARG" ]] && echo true || echo false)"
echo -e "Stereo consistency    : $([[ "$CONSISTENT" == "true" ]] && echo true || echo false)"

echo -e "=======================================\033[0m"
echo ""


# ------------------ Decide whether to run COLMAP ------------------
SKIP_COLMAP=false
EFFECTIVE_DATA_DIR="$DATA_FOLDER"

# Case 1: undist already exists, only use it if not forced to stay in root
if [[ "$FORCE_NO_UNDIST" = false && -d "$DATA_FOLDER/undist" ]]; then
    EFFECTIVE_DATA_DIR="$DATA_FOLDER/undist"
    echo -e "\033[1;34m[SKIP] Found existing undistorted images at $EFFECTIVE_DATA_DIR. Skipping frame extraction, mask extraction, and COLMAP.\033[0m"
    SKIP_COLMAP=true

    # If user provided masks at root and undist/masks does not exist yet, mirror them
    if [[ "$GET_MASKS" == false && "$HAD_INPUT_MASKS" == true && -d "$DATA_FOLDER/masks" && ! -d "$DATA_FOLDER/undist/masks" ]]; then
        mkdir -p "$DATA_FOLDER/undist/masks"
        cp "$DATA_FOLDER/masks"/* "$DATA_FOLDER/undist/masks"/
        echo -e "\033[1;34m[Masks] Mirrored existing root masks into $DATA_FOLDER/undist/masks\033[0m"
    fi
fi

# Case 2: No undist, but sparse exists in root. Use existing sparse and skip COLMAP.
if [[ "$SKIP_COLMAP" = false && -d "$DATA_FOLDER/sparse" ]]; then
    echo -e "\033[1;34m[SKIP] Found existing sparse at $DATA_FOLDER/sparse. Will not run COLMAP or undistortion.\033[0m"
    # If masks are requested but missing, create them. Ensure images exist or extract from video first.
    if $GET_MASKS && [[ ! -d "$DATA_FOLDER/masks" ]]; then
        echo -e "\033[1;34m[Masks] --get_masks requested, masks folder missing. Preparing images then extracting masks.\033[0m"
        ensure_images_ready
        extract_masks
    else
        echo -e "\033[1;34m[Masks] No mask extraction needed\033[0m"
    fi
    EFFECTIVE_DATA_DIR="$DATA_FOLDER"
    SKIP_COLMAP=true
fi

# Case 3: Fresh run. Ensure images, optional masks, then run COLMAP.
# If FORCE_NO_UNDIST is true, run COLMAP without undistortion and keep EFFECTIVE_DATA_DIR = DATA_FOLDER.
# Otherwise run COLMAP with undistortion and set EFFECTIVE_DATA_DIR = DATA_FOLDER/undist.
if [[ "$SKIP_COLMAP" = false ]]; then
    ensure_images_ready
    if $GET_MASKS && [[ ! -d "$DATA_FOLDER/masks" ]]; then
        echo -e "\033[1;34m[Masks] --get_masks requested, masks folder missing. Extracting masks.\033[0m"
        extract_masks
    fi

    if [[ "$FORCE_NO_UNDIST" = true ]]; then
        echo -e "\033[1;34m[COLMAP] Using original dataset folder as effective root (no undistortion)\033[0m"
        run_colmap_pipeline "$DATA_FOLDER" "false"
        EFFECTIVE_DATA_DIR="$DATA_FOLDER"
    else
        run_colmap_pipeline "$DATA_FOLDER" "true"
        EFFECTIVE_DATA_DIR="$DATA_FOLDER/undist"
    fi
fi

echo ""
echo -e "\033[1;38;5;223m======================================="
echo -e "Setup complete"
echo -e "Effective data folder for next steps: $EFFECTIVE_DATA_DIR"

# Check existence of expected subfolders
HAS_IMAGES=false
HAS_SPARSE=false
HAS_MASKS=false

[[ -d "$EFFECTIVE_DATA_DIR/images" ]] && HAS_IMAGES=true
[[ -d "$EFFECTIVE_DATA_DIR/sparse" ]] && HAS_SPARSE=true
[[ -d "$EFFECTIVE_DATA_DIR/masks" ]] && HAS_MASKS=true

# Build the status message
if $HAS_IMAGES && $HAS_SPARSE && $HAS_MASKS; then
    echo -e "\033[1;38;5;223mThe effective data folder contains images, sparse, and masks folders\033[0m"
elif $HAS_IMAGES && $HAS_SPARSE && ! $HAS_MASKS; then
    echo -e "\033[1;38;5;223mThe effective data folder contains images and sparse folders, but no masks folder\033[0m"
elif $HAS_IMAGES && ! $HAS_SPARSE && $HAS_MASKS; then
    echo -e "\033[1;38;5;223mThe effective data folder contains images and masks folders, but no sparse folder\033[0m"
elif ! $HAS_IMAGES && $HAS_SPARSE && $HAS_MASKS; then
    echo -e "\033[1;38;5;223mThe effective data folder contains sparse and masks folders, but no images folder\033[0m"
else
    echo -e "\033[1;38;5;223mThe effective data folder has the following:\033[0m"
    $HAS_IMAGES && echo -e "\033[1;38;5;223m - images ✓" || echo " - images ✗\033[0m"
    $HAS_SPARSE && echo -e "\033[1;38;5;223m - sparse ✓" || echo " - sparse ✗\033[0m"
    $HAS_MASKS && echo -e "\033[1;38;5;223m - masks ✓" || echo " - masks ✗\033[0m"
fi

echo -e "\033[1;38;5;223m=======================================\033[0m"
echo ""


# ------------------ TSDF params ------------------
# Determine the effective dataset dir again in case earlier branches set it.
# Prefer undist if present; else fall back to DATA_FOLDER.
if [[ -z "${EFFECTIVE_DATA_DIR:-}" ]]; then
    if [[ -d "$DATA_FOLDER/undist" ]]; then
        EFFECTIVE_DATA_DIR="$DATA_FOLDER/undist"
    else
        EFFECTIVE_DATA_DIR="$DATA_FOLDER"
    fi
fi

# Locate the COLMAP sparse model directory.
resolve_sparse_dir() {
    local base="$1"
    local sdir=""

    # 1) Common layout: sparse/0
    if [[ -d "$base/sparse/0" ]]; then
        echo "$base/sparse/0"
        return 0
    fi

    # 2) If there's exactly one numeric subdir, use it
    mapfile -t _numdirs < <(find "$base/sparse" -maxdepth 1 -type d -regex '.*/[0-9]+' 2>/dev/null | sort)
    if (( ${#_numdirs[@]} == 1 )); then
        echo "${_numdirs[0]}"
        return 0
    fi

    # 3) If sparse has cameras.bin/poses, use sparse itself
    if compgen -G "$base/sparse/*.bin" >/dev/null || compgen -G "$base/sparse/*.txt" >/dev/null; then
        echo "$base/sparse"
        return 0
    fi

    return 1
}

SPARSE_DIR=""
if SPARSE_DIR="$(resolve_sparse_dir "$EFFECTIVE_DATA_DIR")"; then
    :
# Fallback: try original data folder
elif SPARSE_DIR="$(resolve_sparse_dir "$DATA_FOLDER")"; then
    :
else
    echo "Error: Could not find a valid COLMAP sparse model under:"
    echo "  $EFFECTIVE_DATA_DIR/sparse or $DATA_FOLDER/sparse"
    exit 1
fi

# Decide how to obtain TSDF parameters (voxel_size, depth_trunc)
voxel_size=""
depth_trunc=""
sdf_trunc="5.0"

if [[ -n "$VOXEL_SIZE_OVERRIDE" && -n "$DEPTH_TRUNC_OVERRIDE" ]]; then
    # Both provided by the user: do not run compute_tsdf_params.py
    voxel_size="$VOXEL_SIZE_OVERRIDE"
    depth_trunc="$DEPTH_TRUNC_OVERRIDE"
    echo -e "\033[1;34m[TSDF_PARAMS] Using user provided VOXEL_SIZE=$voxel_size and DEPTH_TRUNC=$depth_trunc (skipping automatic computation)\033[0m"
else
    echo -e "\033[1;34m[TSDF_PARAMS] Computing TSDF parameters using poses at: $SPARSE_DIR\033[0m"

    # Compute TSDF parameters from COLMAP poses and mesh resolution
    require_cmd python
    TSDF_OUT="$(python "$REPO_ROOT/scripts/compute_tsdf_params.py" --colmap_dir "$SPARSE_DIR" --mesh_res "$MESH_RES")" || {
        echo "Error: compute_tsdf_params.py failed."
        exit 1
    }

    # Parse VOXEL_SIZE and DEPTH_TRUNC from the helper's output
    voxel_size="$(grep -Eo 'VOXEL_SIZE=[0-9.eE+-]+' <<< "$TSDF_OUT" | cut -d= -f2)"
    depth_trunc="$(grep -Eo 'DEPTH_TRUNC=[0-9.eE+-]+' <<< "$TSDF_OUT" | cut -d= -f2)"

    if [[ -z "$voxel_size" || -z "$depth_trunc" ]]; then
        echo "Error: Failed to parse TSDF parameters. Output was:"
        echo "$TSDF_OUT"
        exit 1
    fi

    # If the user supplied one of them, override only that one
    if [[ -n "$VOXEL_SIZE_OVERRIDE" ]]; then
        echo -e "\033[1;34m[TSDF_PARAMS] Overriding computed VOXEL_SIZE=$voxel_size with user value $VOXEL_SIZE_OVERRIDE\033[0m"
        voxel_size="$VOXEL_SIZE_OVERRIDE"
    fi
    if [[ -n "$DEPTH_TRUNC_OVERRIDE" ]]; then
        echo -e "\033[1;34m[TSDF_PARAMS] Overriding computed DEPTH_TRUNC=$depth_trunc with user value $DEPTH_TRUNC_OVERRIDE\033[0m"
        depth_trunc="$DEPTH_TRUNC_OVERRIDE"
    fi

    echo -e "\033[1;34m[TSDF_PARAMS] Final VOXEL_SIZE=$voxel_size DEPTH_TRUNC=$depth_trunc SDF_TRUNC=$sdf_trunc\033[0m"
fi


# ================== Initial MVS method execution ==================
conda_env_exists() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: conda not found in PATH."
        exit 1
    fi
    conda env list 2>/dev/null | sed 's/*//g' | awk 'NF{print $1}' | grep -xq "$1"
}

conda_activate() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: conda not found in PATH."
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$1"
}

safe_mv() {
    local src="$1"; local dst="$2"
    if [[ ! -f "$src" ]]; then
        echo "Error: expected output mesh not found: $src"
        exit 1
    fi
    mkdir -p "$(dirname "$dst")"
    mv -f "$src" "$dst"
}

safe_cp_dir() {
    local src="$1"; local dst="$2"
    if [[ ! -d "$src" ]]; then
        echo "Error: source directory not found: $src"
        exit 1
    fi
    if [[ -d "$dst" ]]; then
        echo "Skipping copy. Folder already exists: $dst"
        return 0
    fi
    mkdir -p "$(dirname "$dst")"
    cp -r "$src" "$(dirname "$dst")"
}

if [[ "$MVS_METHOD" == "none" || "$PATH_MVS_METHOD" == "none" ]]; then  # only if initial depths or mesh exist
    if [[ -n "$DEPTH_INIT_DIR" ]]; then
        echo -e "\033[1;34m[MVS] Skipping MVS estimation because initial depth maps are present\033[0m"
    elif [[ -n "$MESH_INIT" ]]; then
        echo -e "\033[1;34m[MESH2DEPTH] Extracting depth from $EFFECTIVE_DATA_DIR/mesh_init.ply to $EFFECTIVE_DATA_DIR/depth_init/\033[0m"
        scripts/mesh_to_depth.sh "$EFFECTIVE_DATA_DIR"
    elif [[ -n "$DEPTH_CUSTOM_DIR" ]]; then
        echo -e "\033[1;34m[DEPTH2MESH2DEPTH] Fusing custom unconsistent depths into initial mesh, then extracting depths to depth_init/\033[0m"
    
        # fuse custom depths into mesh_init.ply
        echo -e "\033[1;34m  [FUSE] Fusing custom depths from $DEPTH_CUSTOM_DIR into initial mesh at $EFFECTIVE_DATA_DIR/mesh_init.ply\033[0m"
        : "${voxel_size:?[FUSE] voxel_size is required but empty}"
        : "${depth_trunc:?[FUSE] depth_trunc is required but empty}"
        : "${sdf_trunc:?[FUSE] sdf_trunc is required but empty}"

        fuse_cmd=(
            scripts/fuse_depth.sh
            "--data_folder=$EFFECTIVE_DATA_DIR"
            "--voxel_size=$voxel_size"
            "--depth_trunc=$depth_trunc"
            "--sdf_trunc=$sdf_trunc"
            )

        if ! "${fuse_cmd[@]}"; then
            echo -e "\033[1;34m[FUSE] Depth fusion failed. If this was a memory issue, try increasing swap memory or reducing --mesh_res (current: $MESH_RES).\033[0m"
        fi

        # change name of post-processed fused mesh to mesh_init.ply  
        DT=$(format5 "${depth_trunc:-UNKNOWN}")
        VS=$(format5 "${voxel_size:-UNKNOWN}")
        ST=$(format5 "${sdf_trunc:-UNKNOWN}") 
        fused_mesh="$EFFECTIVE_DATA_DIR/mesh_trunc${DT}_vox${VS}_sdf${ST}_post.ply"
        if [[ -z "$fused_mesh" ]]; then
            echo "Error: could not find fused mesh output after depth fusion."
            exit 1
        fi
        safe_mv "$fused_mesh" "$EFFECTIVE_DATA_DIR/mesh_init.ply"
        
        # extract depths from fused mesh
        echo -e "\033[1;34m  [MESH2DEPTH] Extracting depths to $EFFECTIVE_DATA_DIR/depth_init/\033[0m"
        scripts/mesh_to_depth.sh "$EFFECTIVE_DATA_DIR"  
    fi
else
    if [[ -z "${PATH_MVS_METHOD:-}" ]]; then
        echo "Error: PATH_MVS_METHOD is not set. Export it to the repo path of $MVS_METHOD."
        exit 1
    fi
    if [[ ! -d "$PATH_MVS_METHOD" ]]; then
        echo "Error: PATH_MVS_METHOD does not exist: $PATH_MVS_METHOD"
        exit 1
    fi

    SCAN_DIR="$EFFECTIVE_DATA_DIR"
    echo -e "\033[1;34m---------------------------------------\033[0m"
    echo -e "\033[1;34m[MVS] Method      : $MVS_METHOD\033[0m"
    echo -e "\033[1;34m[MVS] Repo path   : $PATH_MVS_METHOD\033[0m"
    echo -e "\033[1;34m[MVS] Scan dir    : $SCAN_DIR\033[0m"
    echo -e "\033[1;34m[MVS] voxel_size  : $voxel_size\033[0m"
    echo -e "\033[1;34m[MVS] depth_trunc : $depth_trunc\033[0m"
    echo -e "\033[1;34m---------------------------------------\033[0m"

    # Ensure 4th channel via mask, convert to PNG only when needed
    ensure_rgba_pngs_patch_and_prune_bins() {
        local scan_dir="$1"
        local img_dir="$scan_dir/images"
        local mask_dir="$scan_dir/masks"
        local sparse_root="$scan_dir/sparse"

        # If no masks, skip
        if [[ ! -d "$mask_dir" ]]; then
            echo -e "\033[1;34m[AlphaPNG] 'masks' folder not found; skipping alpha step\033[0m"
            return 0
        fi

        # Tools checks, never abort
        if ! command -v identify >/dev/null 2>&1; then
            echo -e "\033[1;34m[AlphaPNG] 'identify' not found; skipping alpha step\033[0m"
            return 0
        fi
        if ! command -v convert >/dev/null 2>&1; then
            echo -e "\033[1;34m[AlphaPNG] 'convert' not found; skipping alpha step\033[0m"
            return 0
        fi
        command -v colmap >/dev/null 2>&1 || true

        # Collect images
        mapfile -d '' -t IMGS < <(find "$img_dir" -maxdepth 1 -type f \
            \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \) -print0 | sort -z)
        if (( ${#IMGS[@]} == 0 )); then
            echo -e "\033[1;34m[AlphaPNG] No images found in '$img_dir'\033[0m"
            return 0
        fi

        local backup_dir="$img_dir/_orig_converted_to_png"
        local backup_ready=false
        ensure_backup_dir() {
            if [[ "$backup_ready" == false ]]; then
                mkdir -p "$backup_dir"
                backup_ready=true
            fi
        }

        local mapping_file
        mapping_file="$(mktemp -t rename_map.XXXXXX)" || mapping_file=""

        # Find mask for stem
        find_mask_for_stem() {
            local stem="$1" cand
            for cand in "$mask_dir/$stem.png" "$mask_dir/$stem.jpg" "$mask_dir/$stem.jpeg" "$mask_dir/$stem.bmp" "$mask_dir/$stem.tif" "$mask_dir/$stem.tiff"; do
                [[ -f "$cand" ]] && { printf '%s' "$cand"; return 0; }
            done
            return 1
        }

        local converted_ext_changes=0 updated_in_place=0 unchanged=0

        for im in "${IMGS[@]}"; do
            local base stem ext_lc ch mask out_png
            base="$(basename "$im")"
            stem="${base%.*}"
            ext_lc="${base##*.}"; ext_lc="${ext_lc,,}"

            # No mask, leave as is
            if ! mask="$(find_mask_for_stem "$stem")"; then
                ((unchanged++))
                continue
            fi

            ch="$(identify -format "%[channels]" "$im" 2>/dev/null || true)"
            ch="${ch,,}"

            # Already has alpha, leave as is
            if [[ "$ch" == *a* ]]; then
                ((unchanged++))
                continue
            fi

            # Need to append alpha from mask
            case "$ext_lc" in
                png|tif|tiff)
                    if convert "$im" "$mask" -alpha off -compose copy-opacity -composite "$im" 2>/dev/null; then
                        ((updated_in_place++))
                    else
                        echo -e "\033[1;34m[AlphaPNG] Warning: failed to add alpha in place for '$base'\033[0m"
                        ((unchanged++))
                    fi
                    ;;
                jpg|jpeg|bmp)
                    out_png="$img_dir/$stem.png"
                    if convert "$im" "$mask" -alpha off -compose copy-opacity -composite "$out_png" 2>/dev/null; then
                        ensure_backup_dir
                        mv -f "$im" "$backup_dir/$base" 2>/dev/null || true
                        [[ -n "$mapping_file" ]] && echo "$base $stem.png" >> "$mapping_file"
                        ((converted_ext_changes++))
                    else
                        echo -e "\033[1;34m[AlphaPNG] Warning: failed to convert and add alpha for '$base'\033[0m"
                        ((unchanged++))
                    fi
                    ;;
                *)
                    # Try in place, fall back to PNG
                    if convert "$im" "$mask" -alpha off -compose copy-opacity -composite "$im" 2>/dev/null; then
                        ((updated_in_place++))
                    else
                        out_png="$img_dir/$stem.png"
                        if convert "$im" "$mask" -alpha off -compose copy-opacity -composite "$out_png" 2>/dev/null; then
                            ensure_backup_dir
                            mv -f "$im" "$backup_dir/$base" 2>/dev/null || true
                            [[ -n "$mapping_file" ]] && echo "$base $stem.png" >> "$mapping_file"
                            ((converted_ext_changes++))
                        else
                            echo -e "\033[1;34m[AlphaPNG] Warning: failed to handle '$base'\033[0m"
                            ((unchanged++))
                        fi
                    fi
                    ;;
            esac
        done

        echo -e "\033[1;34m[AlphaPNG] Images where alpha added in place: $updated_in_place, Converted to PNG due to alpha need: $converted_ext_changes, Unchanged: $unchanged\033[0m"

        # If no renames, skip patch and prune
        if [[ -z "$mapping_file" || ! -s "$mapping_file" ]]; then
            [[ -n "$mapping_file" ]] && rm -f "$mapping_file"
            # Remove empty backup dir if created but unused
            if [[ -d "$backup_dir" ]] && [[ -z "$(ls -A "$backup_dir" 2>/dev/null)" ]]; then
                rmdir "$backup_dir" 2>/dev/null || true
            fi
            echo -e "\033[1;34m[AlphaPNG] No filename changes; skipping COLMAP patch and BIN pruning\033[0m"
            return 0
        fi

        # Ensure TXT exists
        ensure_txt_for_dir() {
            local dir="$1"
            [[ -d "$dir" ]] || return 0
            if ! compgen -G "$dir/*.txt" >/dev/null; then
                if command -v colmap >/dev/null 2>&1; then
                    echo -e "\033[1;34m[AlphaPNG] Exporting TXT from BIN in '$dir' via colmap model_converter\033[0m"
                    colmap model_converter --input_path "$dir" --output_path "$dir" --output_type TXT >/dev/null 2>&1 || true
                else
                    echo -e "\033[1;34m[AlphaPNG] Warning: No TXT in '$dir' and 'colmap' not found; cannot auto export TXT\033[0m"
                fi
            fi
        }

        patch_images_txt() {
            local file="$1"
            [[ -f "$file" ]] || return 1
            local tmp="${file}.tmp"
            awk -v map="$mapping_file" '
                BEGIN { while ((getline < map) > 0) ren[$1]=$2 }
                {
                    if ($0 ~ /^#/) { print; next }
                    last=$NF
                    if (last in ren) { $NF=ren[last] }
                    print
                }
            ' "$file" > "$tmp" || return 1

            if cmp -s "$file" "$tmp"; then
                rm -f "$tmp"
                return 1
            else
                mv -f "$tmp" "$file"
                return 0
            fi
        }

        prune_bins_tree() {
            local root="$1"
            [[ -d "$root" ]] || return 0
            mapfile -t BINLIST < <(find "$root" -maxdepth 2 -type f -name "*.bin" | sort)
            if (( ${#BINLIST[@]} )); then
                echo -e "\033[1;34m[AlphaPNG] Removing *.bin files under $root:\033[0m"
                printf '  - %s\n' "${BINLIST[@]}"
                rm -f "${BINLIST[@]}" 2>/dev/null || true
            else
                echo -e "\033[1;34m[AlphaPNG] No *.bin files found under $root\033[0m"
            fi
        }

        # Patch images.txt if filenames changed
        local patched_any=false
        if [[ -d "$sparse_root" ]]; then
            ensure_txt_for_dir "$sparse_root"
            if [[ -f "$sparse_root/images.txt" ]]; then
                patch_images_txt "$sparse_root/images.txt" && patched_any=true
            fi
            # Common subfolders
            for d in "$sparse_root/0" "$sparse_root/1"; do
                if [[ -d "$d" ]]; then
                    ensure_txt_for_dir "$d"
                    if [[ -f "$d/images.txt" ]]; then
                        patch_images_txt "$d/images.txt" && patched_any=true
                    fi
                fi
            done
        fi

        rm -f "$mapping_file"

        if $patched_any; then
            echo -e "\033[1;34m[AlphaPNG] images.txt patched due to filename changes. Pruning *.bin to keep TXT only model.\033[0m"
            prune_bins_tree "$sparse_root"
        else
            echo -e "\033[1;34m[AlphaPNG] images.txt not patched. Not removing *.bin\033[0m"
        fi

        # Remove empty backup dir if no files were moved or later cleaned
        if [[ -d "$backup_dir" ]] && [[ -z "$(ls -A "$backup_dir" 2>/dev/null)" ]]; then
            rmdir "$backup_dir" 2>/dev/null || true
        fi

        return 0
    }

    case "$MVS_METHOD" in
        pgsr)
            if [[ -f "$SCAN_DIR/mesh_init.ply" ]]; then
                echo -e "\033[1;34m[SKIP] $SCAN_DIR/mesh_init.ply already exists. Skipping PGSR processing.\033[0m"
            else
                NEED_CONDA=false

                if [[ -d "$SCAN_DIR/pgsr_results" ]]; then
                    echo -e "\033[1;34m[SKIP] Found existing pgsr_results in $SCAN_DIR. Skipping train.py but continuing with render.py\033[0m"
                    NEED_CONDA=true
                    RUN_TRAIN=false
                else
                    echo -e "\033[1;34m[Run] No existing results found. Running train.py...\033[0m"
                    NEED_CONDA=true
                    RUN_TRAIN=true
                fi

                if [[ "$NEED_CONDA" = true ]]; then
                    if ! conda_env_exists "pgsr"; then
                        echo "Error: The conda environment for MVS method 'pgsr' does not exist. It should be named 'pgsr'.\033[0m"
                        exit 1
                    fi

                    conda_activate pgsr
                    pushd "$PATH_MVS_METHOD" >/dev/null

                    if [[ "$RUN_TRAIN" = true ]]; then
                        ensure_rgba_pngs_patch_and_prune_bins "$SCAN_DIR" || true
                        python train.py \
                            -s "$SCAN_DIR" \
                            -m "$SCAN_DIR/pgsr_results" \
                            --max_abs_split_points 0 \
                            --opacity_cull_threshold 0.05 \
                            --quiet \
                            -r 2
                    fi

                    # Render
                    python render.py \
                        -m "$SCAN_DIR/pgsr_results" \
                        --voxel_size "$voxel_size" \
                        --max_depth "$depth_trunc" \
                        --quiet \
                        -r 2 \
                        --num_cluster 1

                    # Move mesh to main folder
                    if [[ -f "$SCAN_DIR/pgsr_results/mesh/tsdf_fusion_post.ply" ]]; then
                        safe_mv "$SCAN_DIR/pgsr_results/mesh/tsdf_fusion_post.ply" "$SCAN_DIR/mesh_init.ply"
                    else
                        echo -e "\033[1;34m[Warning] No tsdf_fusion_post.ply found in $SCAN_DIR/pgsr_results/mesh/\033[0m"
                    fi

                    popd >/dev/null
                    conda_activate "pagas"
                fi
            fi
                    ;;

        2dgs)
            if [[ -f "$SCAN_DIR/mesh_init.ply" ]]; then
                echo -e "\033[1;34m[SKIP] $SCAN_DIR/mesh_init.ply already exists. Skipping 2DGS processing.\033[0m"
            else
                NEED_CONDA=false

                if [[ -d "$SCAN_DIR/2dgs_results" ]]; then
                    echo -e "\033[1;34m[SKIP] Found existing 2dgs_results in $SCAN_DIR. Skipping train.py but continuing with render.py\033[0m"
                    NEED_CONDA=true
                    RUN_TRAIN=false
                else
                    echo -e "\033[1;34m[Run] No existing results found. Running train.py...\033[0m"
                    NEED_CONDA=true
                    RUN_TRAIN=true
                fi

                if [[ "$NEED_CONDA" = true ]]; then
                    # Prefer 'surfel_splatting', else '2dgs', else error
                    ENV_PICKED=""
                    if conda_env_exists "surfel_splatting"; then
                        ENV_PICKED="surfel_splatting"
                    elif conda_env_exists "2dgs"; then
                        ENV_PICKED="2dgs"
                    else
                        echo "Error: The conda environment for MVS method '2dgs' does not exist."
                        echo "Create an environment named 'surfel_splatting' or '2dgs' and try again."
                        exit 1
                    fi

                    conda_activate "$ENV_PICKED"
                    pushd "$PATH_MVS_METHOD" >/dev/null

                    if [[ "$RUN_TRAIN" = true ]]; then
                        ensure_rgba_pngs_patch_and_prune_bins "$SCAN_DIR" || true
                        python train.py \
                            -s "$SCAN_DIR" \
                            -m "$SCAN_DIR/2dgs_results" \
                            -r 2 \
                            --quiet \
                            --test_iterations -1 \
                            --depth_ratio 0.0
                    fi

                    python render.py \
                        -m "$SCAN_DIR/2dgs_results" \
                        --voxel_size "$voxel_size" \
                        --depth_trunc "$depth_trunc" \
                        -r 2 \
                        --num_cluster 1 \
                        --skip_test \
                        --quiet \
                        --depth_ratio 0.0

                    # Move mesh to main folder
                    if [[ -f "$SCAN_DIR/2dgs_results/train/ours_30000/fuse_post.ply" ]]; then
                        safe_mv "$SCAN_DIR/2dgs_results/train/ours_30000/fuse_post.ply" "$SCAN_DIR/mesh_init.ply"
                    else
                        echo -e "\033[1;34m[Warning] No fuse_post.ply found in $SCAN_DIR/2dgs_results/train/ours_30000/"
                    fi

                    popd >/dev/null
                    conda_activate "pagas"
                fi
            fi
                    ;;

        mvsa|mvsa*)
            # Prefer 'mvsanywhere', else 'mvsa', else error
            ENV_PICKED=""
            if conda_env_exists "mvsanywhere"; then
                ENV_PICKED="mvsanywhere"
            elif conda_env_exists "mvsa"; then
                ENV_PICKED="mvsa"
            else
                echo "Error: The conda environment for MVS method 'MVSAnywhere' does not exist."
                echo "Create an environment named 'mvsanywhere' or 'mvsa' and try again."
                exit 1
            fi

            conda_activate "$ENV_PICKED"
            pushd "$PATH_MVS_METHOD" >/dev/null

            DATASET_ROOT="$(dirname "$SCAN_DIR")"
            SCAN_NAME="$(basename "$SCAN_DIR")"

            # Copy sparse into colmap folder
            SPARSE_SRC="${SCAN_DIR}/sparse"
            SPARSE_DST="${SCAN_DIR}/colmap/sparse"
            safe_cp_dir "$SPARSE_SRC" "$SPARSE_DST"

            python ${PATH_MVS_METHOD}/src/mvsanywhere/run_demo.py \
                --name mvsanywhere \
                --output_base_path ${SCAN_DIR}/mvsa_results \
                --config_file ${PATH_MVS_METHOD}/configs/models/mvsanywhere_model.yaml \
                --load_weights_from_checkpoint ${PATH_MVS_METHOD}/weights/mvsanywhere_hero.ckpt \
                --data_config_file ${PATH_MVS_METHOD}/configs/data/colmap/colmap_empty.yaml \
                --scan_parent_directory ${DATASET_ROOT} \
                --scan_name ${SCAN_NAME}:0 \
                --fast_cost_volume \
                --num_workers 8 \
                --batch_size 1 \
                --image_height 480 \
                --image_width 640 \
                --dump_depth_visualization \
                --cache_depths \
                --run_fusion \
                --depth_fuser open3d \
                --fusion_max_depth "$depth_trunc" \
                --fusion_resolution "$voxel_size"

            # Move mesh to main folder
            safe_mv "$SCAN_DIR/mvsa_results/mvsanywhere/colmap/dense_offline/meshes/"$voxel_size"_"$depth_trunc"_open3d/${SCAN_NAME}:0.ply" "$SCAN_DIR/mesh_init.ply"
            popd >/dev/null
            conda_activate "pagas"
            ;;
        *)
            echo "Error: Unsupported MVS method: $MVS_METHOD"
            exit 1
            ;;
    esac
    echo -e "\033[1;34m[MVS] Completed $MVS_METHOD stage. Mesh at: $SCAN_DIR/mesh_init.ply\033[0m"
    echo -e "\033[1;34m[MESH2DEPTH] Extracting depth from $SCAN_DIR/mesh_init.ply to $SCAN_DIR/depth_init/\033[0m"
    scripts/mesh_to_depth.sh $SCAN_DIR
fi


# ------------------ Run PAGaS ------------------
RESULT_DIR="$EFFECTIVE_DATA_DIR/results/depth_init_pagas/depth"
SRC_DIR="$EFFECTIVE_DATA_DIR/depth_init"

# Determine masks name
MASKS_NAME=""
if [[ -d "$EFFECTIVE_DATA_DIR/masks" ]]; then
    MASKS_NAME="masks"
fi

run_pagas() {
  local start_view="$1"
  scripts/run_pagas.sh \
    --data_dir="$EFFECTIVE_DATA_DIR" \
    --depth_folder=depth_init \
    --starting_view="$start_view" \
    --masks_name="$MASKS_NAME" \
    --scale_factors="2 1" \
    --max_steps="200 100" \
    --lr="1e-5 1e-5" \
    --radius_thres="1.42 1.42" \
    --depth_slices="100 100" \
    --normal_reg="0." \
    --num_context_views="10" \
    $EXPOSURE_ARG \
    $SAVE_EXTRA_ARG
}

if [[ ! -d "$RESULT_DIR" ]]; then
  echo -e "\033[1;34m[PAGaS] Refining depth_init into $EFFECTIVE_DATA_DIR/results/depth_init_pagas\033[0m"
  run_pagas 0
else
  # Count files (regular files only) in both folders
  res_count=$(find "$RESULT_DIR" -maxdepth 1 -type f | wc -l | awk '{print $1}')
  src_count=$(find "$SRC_DIR" -maxdepth 1 -type f | wc -l | awk '{print $1}')

  if (( res_count < src_count )); then
    echo -e "\033[1;34m[PAGaS] Refining depth_init into $EFFECTIVE_DATA_DIR/results/depth_init_pagas\033[0m"
    echo -e "\033[1;34m[PAGaS] Resuming refinement at frame $res_count\033[0m"
    run_pagas "$res_count"
  else
    echo -e "\033[1;34m[PAGaS] Refinement appears complete. Skipping.\033[0m"
  fi
fi


# ----------------- Check if mesh already exists -----------------
FUSE_DIR="$EFFECTIVE_DATA_DIR/results/depth_init_pagas"

DT=$(format5 "${depth_trunc:-UNKNOWN}")
VS=$(format5 "${voxel_size:-UNKNOWN}")
ST=$(format5 "${sdf_trunc:-UNKNOWN}")

MESH_BASE="mesh_trunc${DT}_vox${VS}_sdf${ST}"
POST_MESH="$FUSE_DIR/${MESH_BASE}_post.ply"

if [[ -f "$POST_MESH" ]]; then
  echo -e "\033[1;34m[FUSE] Found existing fused mesh: $POST_MESH\033[0m"
  if [[ "$CONSISTENT" == "true" ]]; then
    echo -e "\033[1;34m[StereoFilter] Skipping stereo consistency filtering and depth fusion.\033[0m"
  else
    echo -e "\033[1;34m[FUSE] Skipping depth fusion.\033[0m"
  fi
  echo ""
  echo -e "\033[1;38;5;223m[Automatic pipeline] Completed successfully\033[0m"
  exit 1
fi


# ----------------- Stereo consistency depth filtering -----------------
if [[ "$CONSISTENT" == "true" ]]; then
  CONSIST_DIR="$EFFECTIVE_DATA_DIR/results/depth_init_pagas/depth_consistent"
  DEPTH_DIR="$EFFECTIVE_DATA_DIR/results/depth_init_pagas/depth"
  depth_name="depth_consistent"

  # if consistent folder does not exist or has fewer files than depth, recompute
  if [[ ! -d "$CONSIST_DIR" ]]; then
    echo -e "\033[1;34m[StereoFilter] Applying stereo consistency filtering to refined depths...\033[0m"
    python scripts/consistent_depth.py \
      --data_folder="$EFFECTIVE_DATA_DIR/results/depth_init_pagas"
  else
    res_count=$(find "$CONSIST_DIR" -maxdepth 1 -type f | wc -l | awk '{print $1}')
    src_count=$(find "$DEPTH_DIR" -maxdepth 1 -type f | wc -l | awk '{print $1}')

    if (( res_count < src_count )); then
      echo -e "\033[1;34m[StereoFilter] Incomplete consistent depths detected. Recomputing...\033[0m"
      python scripts/consistent_depth.py \
        --data_folder="$EFFECTIVE_DATA_DIR/results/depth_init_pagas"
    else
      echo -e "\033[1;34m[StereoFilter] Consistent depths already complete. Skipping.\033[0m"
    fi
  fi
else
  depth_name="depth"
fi


# ------------------ Fuse refined depths into a mesh ------------------
echo -e "\033[1;34m[FUSE] Depth fusion of refined depths using:\033[0m"
echo -e "\033[1;34m       depth_trunc=$DT, voxel_size=$VS, sdf_trunc=$ST\033[0m"

: "${voxel_size:?[FUSE] voxel_size is required but empty}"
: "${depth_trunc:?[FUSE] depth_trunc is required but empty}"
: "${sdf_trunc:?[FUSE] sdf_trunc is required but empty}"

fuse_cmd=(
scripts/fuse_depth.sh
"--data_folder=$FUSE_DIR"
"--depth_name=$depth_name"
"--voxel_size=$voxel_size"
"--depth_trunc=$depth_trunc"
"--sdf_trunc=$sdf_trunc"
)

if ! "${fuse_cmd[@]}"; then
echo -e "\033[1;34m[FUSE] Depth fusion failed. If this was a memory issue, try increasing swap memory or reducing --mesh_res (current: $MESH_RES)\033[0m"
fi


echo ""
echo -e "\033[1;38;5;223m[Automatic pipeline] Completed successfully\033[0m"
