#!/usr/bin/env bash
# License: CC BY-NC 4.0 - see /LICENSE
set -euo pipefail
ulimit -c 0 2>/dev/null || true

# ============================================================
# COLMAP SfM (+ optional MVS)
# Pose-first, fast defaults; no sparse densification unless MVS
# Masks are USED FOR SFM ONLY if --masks is explicitly provided.
# If --undistort_images is given, masks are ALSO undistorted into undist/masks
# whenever a masks/ folder exists and has files (even without --masks).
# ============================================================

usage() {
  cat <<EOF
Usage:
  $(basename "$0") DATASET_DIR
    [--intrinsics REL_PATH]
    [--undistort_images]
    [--shared_intrinsics]
    [--masks]
    [--mvs | --mvs_ultra]
    [--sequential]
    [--seq_overlap N]
    [--vocab_tree [PATH]]

Env toggles:
  WANT_HEAVY_SIFT=1   # enable Affine + DSP SIFT (slower)
  COVER_SKIP_PCT=88   # skip heavy R3 if >= this % registered
  SPARSE_DENSIFY=1    # run point_triangulator even without MVS (default off)
  TO_SEQ_MIN=25 TO_MAPPER_MIN=30 TO_MM_MAPPER_MIN=40 TO_HEAVY_MIN=45  # timeouts (min)
EOF
  exit 1
}

# ---------------------------
# Parse args
# ---------------------------
DATASET_DIR=""
INTR_REL=""
UNDISTORT_IMAGES="false"
SHARED_INTR="false"
RUN_MVS_ULTRA="false"
USE_SEQUENTIAL="false"
SEQ_OVERLAP="8"
USE_VOCAB_TREE="false"
VOCAB_TREE=""
USE_MASKS_FLAG="false"

[[ $# -ge 1 ]] || usage
DATASET_DIR="$1"; shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --intrinsics)         INTR_REL="${2:-}"; [[ -n "$INTR_REL" ]] || usage; shift 2 ;;
    --undistort_images)   UNDISTORT_IMAGES="true"; shift ;;
    --shared_intrinsics)  SHARED_INTR="true"; shift ;;
    --masks)              USE_MASKS_FLAG="true"; shift ;;
    --mvs)                RUN_MVS_ULTRA="true"; shift ;;
    --mvs_ultra)          RUN_MVS_ULTRA="true"; shift ;;
    --sequential)         USE_SEQUENTIAL="true"; shift ;;
    --seq_overlap)        SEQ_OVERLAP="${2:-}"; [[ "$SEQ_OVERLAP" =~ ^[0-9]+$ ]] || usage; shift 2 ;;
    --vocab_tree)
      USE_VOCAB_TREE="true"
      if [[ $# -ge 2 && ! "${2:-}" =~ ^- ]]; then VOCAB_TREE="${2:-}"; shift 2; else shift 1; fi
      ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

# ---------------------------
# Paths & setup
# ---------------------------
IMAGES_DIR="$DATASET_DIR/images"
DB_P1="$DATASET_DIR/database_p1.db"
DB_R2="$DATASET_DIR/database_r2.db"
DB_R3="$DATASET_DIR/database_r3.db"
SPARSE_DIR="$DATASET_DIR/sparse"
SPARSE0_DIR="$SPARSE_DIR/0"
UNDIST_DIR="$DATASET_DIR/undist"
MVS_DIR="$DATASET_DIR/mvs"
TMP_LOGS="$DATASET_DIR/.logs_tmp"
MASKS_DIR="$DATASET_DIR/masks"

[[ -d "$IMAGES_DIR" ]] || { echo "Missing $IMAGES_DIR"; exit 1; }
mkdir -p "$SPARSE_DIR" "$TMP_LOGS"
cleanup_tmp_logs(){ rm -rf "$TMP_LOGS" 2>/dev/null || true; }
trap cleanup_tmp_logs EXIT
rm -f "$DB_P1" "$DB_R2" "$DB_R3"

# ----- CUDA env -----
fix_cuda_env() {
  local ld="${LD_LIBRARY_PATH:-}"
  local pruned
  pruned="$(echo "$ld" | tr ':' '\n' | grep -vE '/cuda[^/]*/lib(64)?/stubs/?$' | paste -sd: -)"
  if [[ -d /usr/lib/x86_64-linux-gnu ]]; then export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${pruned:+:$pruned}";
  else export LD_LIBRARY_PATH="${pruned:-}"; fi
}
cuda_preflight() {
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    local cleaned=""; IFS=':' read -ra parts <<< "$LD_LIBRARY_PATH"
    for p in "${parts[@]}"; do [[ "$p" =~ /cuda.*/stubs ]] && continue; [[ -d "$p" ]] && cleaned="${cleaned:+$cleaned:}$p"; done
    export LD_LIBRARY_PATH="$cleaned"
  fi
  if ! ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1'; then
    for d in /usr/lib/x86_64-linux-gnu /usr/lib/wsl/lib /usr/lib; do
      [[ -d "$d" ]] || continue
      if ls "$d"/libcuda.so.* >/dev/null 2>&1; then export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; break; fi
    done
  fi
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
}
cuda_device_count() {
  local n=""
  if command -v python3 >/dev/null 2>&1; then
    n="$(python3 - <<'PY'
import ctypes
try:
    lib = ctypes.CDLL('libcuda.so.1')
    cuInit = lib.cuInit; cuInit.argtypes=[ctypes.c_uint]; cuInit.restype=ctypes.c_int
    cuDeviceGetCount = lib.cuDeviceGetCount; cuDeviceGetCount.argtypes=[ctypes.POINTER(ctypes.c_int)]; cuDeviceGetCount.restype=ctypes.c_int
    if cuInit(0)!=0: print(0)
    else:
        cnt=ctypes.c_int(0)
        print(cnt.value if cuDeviceGetCount(ctypes.byref(cnt))!=0 else cnt.value)
except Exception: print(0)
PY
)"; fi
  if [[ -z "$n" || "$n" == "0" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      n="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c . || echo 0)"
    fi
  fi
  echo "${n:-0}"
}
fix_cuda_env
cuda_preflight
GPU_CNT="$(cuda_device_count)"
echo ">> CUDA devices visible: $GPU_CNT"

SIFT_GPU=( ); MATCH_GPU=( )
if [[ "${GPU_CNT:-0}" -gt 0 ]]; then SIFT_GPU=( --SiftExtraction.use_gpu 1 ); MATCH_GPU=( --SiftMatching.use_gpu 1 ); else SIFT_GPU=( --SiftExtraction.use_gpu 0 ); MATCH_GPU=( --SiftMatching.use_gpu 0 ); fi

# ----- COLMAP capabilities -----
COLMAP_VER_LINE="$(colmap 2>&1 | head -n1 || true)"; [[ -n "${COLMAP_VER_LINE:-}" ]] || COLMAP_VER_LINE="COLMAP (version unknown)"
echo ">> Using COLMAP: ${COLMAP_VER_LINE}"
CAP_ROOTSIFT=0; CAP_AFFINE=0; CAP_DSP=0; CAP_GUIDED=0; CAP_SPATIAL=0; CAP_VOCAB=0
{ out="$(colmap feature_extractor --help 2>&1 || true)"; [[ "$out" == *"root_sift"* ]] && CAP_ROOTSIFT=1; [[ "$out" == *"estimate_affine_shape"* ]] && CAP_AFFINE=1; [[ "$out" == *"domain_size_pooling"* ]] && CAP_DSP=1; } >/dev/null 2>&1 || true
{ out="$(colmap exhaustive_matcher --help 2>&1 || true)"; [[ "$out" == *"guided_matching"* ]] && CAP_GUIDED=1; } >/dev/null 2>&1 || true
{ out="$(colmap help 2>&1 || true)"; [[ "$out" =~ (^|[[:space:]])spatial_matcher($|[[:space:]]) ]] && CAP_SPATIAL=1; [[ "$out" =~ (^|[[:space:]])vocab_tree_matcher($|[[:space:]]) ]] && CAP_VOCAB=1; } >/dev/null 2>&1 || true
echo ">> Capabilities: root_sift=$CAP_ROOTSIFT, affine=$CAP_AFFINE, dsp=$CAP_DSP, guided=$CAP_GUIDED, spatial=$CAP_SPATIAL, vocab=$CAP_VOCAB"

SIFT_SAFE_OPTS=()
(( CAP_ROOTSIFT )) && SIFT_SAFE_OPTS+=( --SiftExtraction.root_sift 1 )
if [[ "${WANT_HEAVY_SIFT:-0}" == "1" ]]; then
  (( CAP_AFFINE )) && SIFT_SAFE_OPTS+=( --SiftExtraction.estimate_affine_shape 1 )
  (( CAP_DSP    )) && SIFT_SAFE_OPTS+=( --SiftExtraction.domain_size_pooling 1 )
fi
MATCH_SAFE_OPTS=()
(( CAP_GUIDED )) && MATCH_SAFE_OPTS+=( --SiftMatching.guided_matching 1 )

# ----- Timeouts -----
: "${TO_SEQ_MIN:=25}"; : "${TO_MAPPER_MIN:=30}"; : "${TO_MM_MAPPER_MIN:=40}"; : "${TO_HEAVY_MIN:=45}"

# ----- Logging helpers -----
qr(){ local log="$1"; shift; ( "$@" > "$log" 2>&1 ); local ec=$?; if [[ $ec -ne 0 ]]; then echo "!! Command failed. See log: $log"; { echo "----- tail of $log -----"; tail -n 160 "$log" 2>/dev/null || true; echo "--------------------------"; } >&2; exit $ec; fi; }
qrs(){ local log="$1"; shift; ( "$@" > "$log" 2>&1 ); echo "[exit_code=$?]" >> "$log"; }
have_timeout(){ command -v timeout >/dev/null 2>&1; }
tqrs(){ local log="$1" to="$2"; shift 2; if have_timeout; then ( timeout "${to}m" "$@" > "$log" 2>&1 ); echo "[exit_code=$?]" >> "$log"; else qrs "$log" "$@"; fi; }

# ----- Small utils -----
ensure_sparse_logs(){ [[ -d "$SPARSE0_DIR" ]] && mkdir -p "$SPARSE0_DIR/logs" || true; }
mark_model_origin(){ echo "$(basename "$2")" > "$1/.db_origin"; }
same_origin(){ [[ -f "$1/.db_origin" ]] && grep -qx "$(basename "$2")" "$1/.db_origin"; }
db_for_current_model(){ [[ -f "$SPARSE0_DIR/.db_origin" ]] && [[ -f "$DATASET_DIR/$(cat "$SPARSE0_DIR/.db_origin")" ]] && echo "$DATASET_DIR/$(cat "$SPARSE0_DIR/.db_origin")" || echo ""; }

count_all_input_images(){
  find "$IMAGES_DIR" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png'  -o \
       -iname '*.bmp' -o -iname '*.tif'  -o -iname '*.tiff' -o \
       -iname '*.pgm' -o -iname '*.ppm' \) | wc -l | tr -d ' '
}
count_registered_images() {
  local d="$1" n=""
  if [[ -f "$d/images.txt" ]]; then
    n="$(sed -nE 's/^# Number of images:[[:space:]]*([0-9]+).*/\1/p' "$d/images.txt" | head -n1 || true)"
    if [[ -z "$n" ]]; then
      n="$(awk '
        /^#/ {next}
        NF>0 {nonc++; if (nonc % 2 == 1) imgs++}
        END {print imgs+0}
      ' "$d/images.txt" 2>/dev/null || echo 0)"
    fi
  else n=0; fi
  echo "${n:-0}"
}

# ---------------------------
# Masks (optional for SfM via --masks)
# ---------------------------
USE_MASKS="false"; MASK_ARGS=()
if [[ "$USE_MASKS_FLAG" == "true" ]]; then
  if [[ ! -d "$MASKS_DIR" ]]; then
    echo "Error: --masks specified but $MASKS_DIR does not exist"
    exit 1
  fi
  shopt -s nullglob nocaseglob
  mask_files=( "$MASKS_DIR"/*.jpg "$MASKS_DIR"/*.jpeg "$MASKS_DIR"/*.png "$MASKS_DIR"/*.bmp "$MASKS_DIR"/*.tif "$MASKS_DIR"/*.tiff "$MASKS_DIR"/*.pgm "$MASKS_DIR"/*.ppm )
  shopt -u nullglob nocaseglob
  if (( ${#mask_files[@]} == 0 )); then
    echo "Error: --masks specified but $MASKS_DIR is empty"
    exit 1
  fi
  USE_MASKS="true"
  MASK_ARGS=( --ImageReader.mask_path "$MASKS_DIR" )
  echo ">> Using masks for SfM from: $MASKS_DIR"
fi

# ---------------------------
# Intrinsics
# ---------------------------
INTR_TXT=""
if [[ -n "$INTR_REL" ]]; then
  CAND="$DATASET_DIR/$INTR_REL"; INTR_TXT="$CAND"
  [[ -d "$CAND" ]] && INTR_TXT="$CAND/cameras.txt"
  [[ -f "$INTR_TXT" ]] || { echo "Intrinsics cameras.txt not found at: $INTR_TXT"; exit 1; }
fi
CAM_MODEL=""; CAM_PARAMS_SPACE=""
if [[ -n "$INTR_TXT" ]]; then
  read -r CAM_MODEL CAM_PARAMS_SPACE <<<"$(
    awk '!/^#/ && NF>0 { m=$2; p=""; for(i=5;i<=NF;i++){ p=(p==""?$i:p" "$i) } print m, p; exit }' "$INTR_TXT"
  )" || true
  [[ -n "$CAM_MODEL" ]] || { echo "Could not parse intrinsics from $INTR_TXT"; exit 1; }
fi

UNKNOWN_INTR=$([[ -z "$INTR_TXT" ]] && echo "1" || echo "0")
if [[ "$UNKNOWN_INTR" == "1" ]]; then
  if [[ "$UNDISTORT_IMAGES" == "true" ]]; then
    CAM_FOR_READER="OPENCV";   BA_REFINE_F=1; BA_REFINE_PP=0; BA_REFINE_EXTRA=1
  else
    CAM_FOR_READER="PINHOLE";  BA_REFINE_F=1; BA_REFINE_PP=0; BA_REFINE_EXTRA=0
  fi
  SINGLE_CAMERA=$([[ "$SHARED_INTR" == "true" ]] && echo "1" || echo "0")
else
  CAM_FOR_READER="${CAM_MODEL}"; SINGLE_CAMERA="1"
  BA_REFINE_F=0; BA_REFINE_PP=0; BA_REFINE_EXTRA=0
fi

READER_OPTS=( --ImageReader.single_camera "$SINGLE_CAMERA" --ImageReader.camera_model "$CAM_FOR_READER" --ImageReader.default_focal_length_factor 1.2 )
if [[ -n "$INTR_TXT" ]]; then
  CAM_PARAMS_CSV="$(echo "$CAM_PARAMS_SPACE" | tr -s ' ' ',' )"
  [[ -n "$CAM_PARAMS_CSV" ]] && READER_OPTS+=( --ImageReader.camera_params "$CAM_PARAMS_CSV" )
fi
READER_OPTS+=( "${MASK_ARGS[@]}" )

# ---------------------------
# Robust model promotion
# ---------------------------
promote_model_safely() {
  shopt -s nullglob
  local best_dir="" best_imgs=-1
  for D in "$SPARSE_DIR"/*; do
    [[ -d "$D" ]] || continue
    local b="$(basename "$D")"; [[ "$b" =~ ^[0-9]+$ ]] || continue
    [[ -f "$D/images.txt" ]] || colmap model_converter --input_path "$D" --output_path "$D" --output_type TXT >/dev/null 2>&1 || true
    local n="$(count_registered_images "$D")"
    if (( n > best_imgs )); then best_imgs="$n"; best_dir="$D"; fi
  done
  shopt -u nullglob
  [[ -z "$best_dir" ]] && return 0

  local cur_imgs=-1
  if [[ -d "$SPARSE0_DIR" ]]; then cur_imgs="$(count_registered_images "$SPARSE0_DIR")"; else cur_imgs=-1; fi

  if (( best_imgs > cur_imgs )); then
    if [[ "$best_dir" != "$SPARSE0_DIR" ]]; then
      rm -rf "$SPARSE0_DIR"
      mv "$best_dir" "$SPARSE0_DIR"
    fi
    find "$SPARSE_DIR" -maxdepth 1 -mindepth 1 -type d -regex '.*/[0-9]+' -not -path "$SPARSE0_DIR" -exec rm -rf {} + 2>/dev/null || true
    colmap model_converter --input_path "$SPARSE0_DIR" --output_path "$SPARSE0_DIR" --output_type TXT >/dev/null 2>&1 || true
    colmap model_converter --input_path "$SPARSE0_DIR" --output_path "$SPARSE0_DIR/sparse.ply" --output_type PLY >/dev/null 2>&1 || true
  fi
  ensure_sparse_logs
}

attach_with_current_origin() {
  local tag="$1"; local db_cur; db_cur="$(db_for_current_model)"; [[ -n "$db_cur" ]] || return 0
  echo ">> $tag: image_registrator (origin DB)"
  qrs "$SPARSE0_DIR/logs/${tag}_image_registrator.log" \
    colmap image_registrator \
      --database_path "$db_cur" \
      --input_path  "$SPARSE0_DIR" \
      --output_path "$SPARSE0_DIR" \
      --Mapper.abs_pose_min_num_inliers 12 \
      --Mapper.abs_pose_max_error 8 \
      --Mapper.init_min_num_inliers 45 \
      --Mapper.ba_refine_focal_length      $BA_REFINE_F \
      --Mapper.ba_refine_principal_point   $BA_REFINE_PP \
      --Mapper.ba_refine_extra_params      $BA_REFINE_EXTRA
  qrs "$SPARSE0_DIR/logs/${tag}_bundle_after_reg.log" \
    colmap bundle_adjuster \
      --input_path  "$SPARSE0_DIR" \
      --output_path "$SPARSE0_DIR" \
      --BundleAdjustment.refine_extrinsics 1 \
      --BundleAdjustment.refine_focal_length $BA_REFINE_F \
      --BundleAdjustment.refine_principal_point 0 \
      --BundleAdjustment.refine_extra_params $BA_REFINE_EXTRA
  colmap model_converter --input_path "$SPARSE0_DIR" --output_path "$SPARSE0_DIR" --output_type TXT >/dev/null 2>&1 || true
  colmap model_converter --input_path "$SPARSE0_DIR" --output_path "$SPARSE0_DIR/sparse.ply" --output_type PLY >/dev/null 2>&1 || true
}

# ---------- Anchor & mask helpers ----------
MASK_COARSE_NEIGHBORS="${MASK_COARSE_NEIGHBORS:-10}"

build_masked_neighbor_pairs() {
  local out_pairs="$1" k="$2"
  ensure_sparse_logs
  (
    shopt -s nullglob nocaseglob
    for f in "$IMAGES_DIR"/*; do
      case "${f,,}" in *.jpg|*.jpeg|*.png|*.bmp|*.tif|*.tiff|*.pgm|*.ppm) basename "$f";; esac
    done | sort -V
    shopt -u nullglob nocaseglob
  ) > "$SPARSE0_DIR/logs/_all_images.txt"

  awk 'BEGIN{IGNORECASE=1}
    /^[[:space:]]*[0-9]+[[:space:]]/ {
      name=""; for (i=10; i<=NF; i++) { name = (name ? name " " : "") $i }
      if (name ~ /\.(jpg|jpeg|png|bmp|tif|tiff|pgm|ppm)$/) print name
    }' "$SPARSE0_DIR/images.txt" | sort -u > "$SPARSE0_DIR/logs/_registered.txt"

  if grep -q '[[:space:]]' "$SPARSE0_DIR/logs/_registered.txt"; then
    echo ">> WARNING: image filenames contain spaces; pair lists may break." >&2
  fi

  awk -v K="$k" -v regfile="$SPARSE0_DIR/logs/_registered.txt" '
    BEGIN{ while((getline<regfile)>0){regset[$0]=1} }
    { all[ni]=$0; ni++ }
    END{
      for(i=0;i<ni;i++){
        name=all[i]
        if(!(name in regset)){
          cnt=0
          for(d=1; d<=ni && cnt<K; d++){
            left=i-d; right=i+d
            if(left>=0  && (all[left]  in regset) && cnt<K){ printf "%s %s\n", name, all[left];  cnt++ }
            if(right<ni && (all[right] in regset) && cnt<K){ printf "%s %s\n", name, all[right]; cnt++ }
          }
        }
      }
    }
  ' "$SPARSE0_DIR/logs/_all_images.txt" > "$out_pairs"
}

coarse_mask_attach_if_possible() {
  [[ "$USE_MASKS" == "true" ]] || return 0
  [[ -d "$SPARSE0_DIR" && -f "$SPARSE0_DIR/images.txt" ]] || return 0
  local reg_now total_in; reg_now=$(count_registered_images "$SPARSE0_DIR"); total_in=$(count_all_input_images)
  (( reg_now < total_in )) || return 0
  local DB_CUR; DB_CUR="$(db_for_current_model)"; [[ -f "$DB_CUR" ]] || { echo ">> Mask-coarse: no DB bound; skip."; return 0; }
  ensure_sparse_logs
  local pairs="$SPARSE0_DIR/logs/mask_coarse_pairs.txt"
  echo ">> Mask-coarse: building neighbor pairs (K=$MASK_COARSE_NEIGHBORS)"
  build_masked_neighbor_pairs "$pairs" "$MASK_COARSE_NEIGHBORS"
  [[ -s "$pairs" ]] || { echo ">> Mask-coarse: no pairs; skip."; return 0; }
  qrs "$SPARSE0_DIR/logs/Rmask_matches_importer.log" \
    colmap matches_importer \
      --database_path "$DB_CUR" \
      --match_list_path "$pairs" \
      --match_type pairs \
      "${MATCH_GPU[@]}" \
      --SiftMatching.cross_check 0 \
      --SiftMatching.max_ratio 0.95 \
      --SiftMatching.max_num_matches 70000
  attach_with_current_origin "Rmask_attach"
}

build_anchor_pairs(){
  local out_pairs="$1" window="${2:-16}"
  ensure_sparse_logs
  (
    shopt -s nullglob nocaseglob
    for f in "$IMAGES_DIR"/*; do
      case "${f,,}" in *.jpg|*.jpeg|*.png|*.bmp|*.tif|*.tiff|*.pgm|*.ppm) basename "$f";; esac
    done | sort -V
    shopt -u nullglob nocaseglob
  ) > "$SPARSE0_DIR/logs/_all_images.txt"
  local N; N=$(wc -l < "$SPARSE0_DIR/logs/_all_images.txt" | tr -d ' ')
  : "${ANCHOR_STRIDE:=5}"; : "${ANCHOR_WIN:=$window}"
  awk -v WIN="$ANCHOR_WIN" -v STRIDE="$ANCHOR_STRIDE" -v N="$N" '
    { imgs[NR-1]=$0 }
    END{
      for(a=0; a<N; a+=STRIDE){
        for(d=1; d<=WIN; d++){
          i=a; j=a+d; if(j<N){ printf "%s %s\n", imgs[i], imgs[j] }
        }
      }
    }' "$SPARSE0_DIR/logs/_all_images.txt" | awk '!seen[$0]++' > "$out_pairs"
}
anchor_sweep_force_pairs(){
  local tag="$1" window="${2:-16}"
  ensure_sparse_logs
  local db_cur; db_cur="$(db_for_current_model)"; [[ -n "$db_cur" ]] || return 0
  local pairs="$SPARSE0_DIR/logs/${tag}_pairs.txt"
  echo ">> ${tag}: generating forced neighbor pairs (window=${window})"
  build_anchor_pairs "$pairs" "$window"
  [[ -s "$pairs" ]] || { echo ">> ${tag}: no forced pairs built"; return 0; }
  qrs "$SPARSE0_DIR/logs/${tag}_matches_importer.log" \
    colmap matches_importer \
      --database_path "$db_cur" \
      --match_list_path "$pairs" \
      --match_type pairs \
      "${MATCH_GPU[@]}" \
      --SiftMatching.cross_check 0 \
      --SiftMatching.max_ratio 0.95 \
      --SiftMatching.max_num_matches 70000
}
print_sparse_summary() {
  local total reg pct
  total="$(count_all_input_images)"
  if [[ -d "$SPARSE0_DIR" ]]; then
    reg="$(count_registered_images "$SPARSE0_DIR")"
  else
    reg=0
  fi
  if [[ "${total:-0}" -gt 0 ]]; then
    pct=$(( 100 * reg / total ))
  else
    pct=0
  fi
  echo ">> Sparse summary: poses captured = ${reg} / ${total} (${pct}%)"
}

ensure_sparse_zero_layout() {
  # $1 = path to a COLMAP sparse dir (e.g., "$UNDIST_DIR/sparse")
  local SD="$1"
  [[ -d "$SD" ]] || return 0
  if [[ ! -d "$SD/0" ]]; then
    mkdir -p "$SD/0"
    shopt -s nullglob
    for f in cameras.bin images.bin points3D.bin cameras.txt images.txt points3D.txt; do
      if [[ -f "$SD/$f" ]]; then
        mv "$SD/$f" "$SD/0/$f"
        ln -s "0/$f" "$SD/$f"
      fi
    done
    shopt -u nullglob
  fi
}


# ============================================================
# PASS 1: features + (optional) sequential + (optional) exhaustive
# ============================================================
echo ">> Feature extraction (pass 1)"
qr "$TMP_LOGS/01_feature_extractor_p1.log" \
  colmap feature_extractor \
    --database_path "$DB_P1" \
    --image_path    "$IMAGES_DIR" \
    "${READER_OPTS[@]}" \
    "${SIFT_GPU[@]}" \
    "${SIFT_SAFE_OPTS[@]}" \
    --SiftExtraction.max_num_features 26000 \
    --SiftExtraction.peak_threshold 0.01

if [[ "$USE_SEQUENTIAL" == "true" ]]; then
  echo ">> Sequential matching (seed neighbor graph, overlap=$SEQ_OVERLAP)"
  tqrs "$TMP_LOGS/02a_sequential_matcher_seed.log" "$TO_SEQ_MIN" \
    colmap sequential_matcher \
      --database_path "$DB_P1" \
      "${MATCH_GPU[@]}" \
      --SiftMatching.cross_check 0 \
      "${MATCH_SAFE_OPTS[@]}" \
      --SequentialMatching.overlap "$SEQ_OVERLAP" \
      --SequentialMatching.quadratic_overlap 1
fi

if [[ "$USE_SEQUENTIAL" != "true" ]]; then
  echo ">> Exhaustive matching (pass 1)"
  tqrs "$TMP_LOGS/02c_exhaustive_matcher_p1.log" "$TO_SEQ_MIN" \
    colmap exhaustive_matcher \
      --database_path "$DB_P1" \
      "${MATCH_GPU[@]}" \
      --SiftMatching.cross_check 1 \
      "${MATCH_SAFE_OPTS[@]}"
else
  echo ">> Skipping exhaustive pass because --sequential was requested"
fi

echo ">> Incremental mapping (pass 1)"
tqrs "$TMP_LOGS/03_mapper_p1.log" "$TO_MAPPER_MIN" \
  colmap mapper \
    --database_path "$DB_P1" \
    --image_path    "$IMAGES_DIR" \
    --output_path   "$SPARSE_DIR" \
    --Mapper.init_min_num_inliers 60 \
    --Mapper.init_min_tri_angle 8 \
    --Mapper.init_max_error 6 \
    --Mapper.abs_pose_min_num_inliers 25 \
    --Mapper.ba_refine_focal_length      $BA_REFINE_F \
    --Mapper.ba_refine_principal_point   $BA_REFINE_PP \
    --Mapper.ba_refine_extra_params      $BA_REFINE_EXTRA \
    --Mapper.ba_global_max_num_iterations 120 \
    --Mapper.ba_local_max_num_iterations 80 \
    --Mapper.multiple_models 0 \
    --Mapper.max_num_models 1

promote_model_safely
mark_model_origin "$SPARSE0_DIR" "$DB_P1"
mkdir -p "$SPARSE0_DIR/logs"; mv "$TMP_LOGS"/* "$SPARSE0_DIR/logs/" || true

echo ">> Status: registered=$(count_registered_images "$SPARSE0_DIR") / total=$(count_all_input_images)"
TOTAL_IN=$(count_all_input_images)
REG_NOW=$(count_registered_images "$SPARSE0_DIR")

# ---------- Attach leftovers with origin DB ----------
if (( REG_NOW < TOTAL_IN )); then
  attach_with_current_origin "R0_attach_p1"
  REG_NOW=$(count_registered_images "$SPARSE0_DIR")
fi

# ---------- Rescue 1 ----------
if (( REG_NOW < TOTAL_IN )); then
  echo ">> Rescue 1: more permissive matching and mapping"
  tqrs "$SPARSE0_DIR/logs/R1_matcher.log" "$TO_SEQ_MIN" \
    colmap exhaustive_matcher \
      --database_path "$DB_P1" \
      "${MATCH_GPU[@]}" \
      --SiftMatching.cross_check 1 \
      "${MATCH_SAFE_OPTS[@]}" \
      --SiftMatching.max_num_matches 50000 \
      --SiftMatching.max_ratio 0.92
  attach_with_current_origin "R1_attach"
  REG_NOW=$(count_registered_images "$SPARSE0_DIR")
fi

# Mask-guided attach (only if --masks used)
if (( REG_NOW < TOTAL_IN )) && [[ "$USE_MASKS" == "true" ]]; then
  echo ">> Mask-guided neighbor seed + attach (post R1)"
  coarse_mask_attach_if_possible
  REG_NOW=$(count_registered_images "$SPARSE0_DIR")
fi

# ---------- Rescue 2 ----------
if (( REG_NOW < TOTAL_IN )) && [[ "$UNKNOWN_INTR" == "1" && "$UNDISTORT_IMAGES" == "false" ]]; then
  echo ">> Rescue 2: distortion-aware features (OPENCV)"
  tqrs "$SPARSE0_DIR/logs/R2_feature_extractor.log" "$TO_SEQ_MIN" \
    colmap feature_extractor \
      --database_path "$DB_R2" \
      --image_path    "$IMAGES_DIR" \
      --ImageReader.single_camera "$SINGLE_CAMERA" \
      --ImageReader.camera_model OPENCV \
      --ImageReader.default_focal_length_factor 1.2 \
      "${MASK_ARGS[@]}" \
      "${SIFT_GPU[@]}" \
      "${SIFT_SAFE_OPTS[@]}" \
      --SiftExtraction.max_num_features 26000 \
      --SiftExtraction.peak_threshold 0.01
  tqrs "$SPARSE0_DIR/logs/R2_matcher.log" "$TO_SEQ_MIN" \
    colmap exhaustive_matcher \
      --database_path "$DB_R2" \
      "${MATCH_GPU[@]}" \
      --SiftMatching.cross_check 1 \
      "${MATCH_SAFE_OPTS[@]}"
  tqrs "$SPARSE0_DIR/logs/R2_mapper.log" "$TO_MAPPER_MIN" \
    colmap mapper \
      --database_path "$DB_R2" \
      --image_path    "$IMAGES_DIR" \
      --output_path   "$SPARSE_DIR" \
      --Mapper.init_min_num_inliers 45 \
      --Mapper.init_min_tri_angle 4 \
      --Mapper.init_max_error 7 \
      --Mapper.abs_pose_min_num_inliers 18 \
      --Mapper.ba_refine_focal_length 1 \
      --Mapper.ba_refine_principal_point 0 \
      --Mapper.ba_refine_extra_params 1 \
      --Mapper.multiple_models 0 \
      --Mapper.max_num_models 1
  promote_model_safely
  mark_model_origin "$SPARSE0_DIR" "$DB_R2"
  REG_NOW=$(count_registered_images "$SPARSE0_DIR")
fi

# ---------- Rescue 3 (heavy) ----------
COVER_SKIP_PCT="${COVER_SKIP_PCT:-88}"
if (( REG_NOW < TOTAL_IN )); then
  COVER_PCT=$(( 100 * REG_NOW / TOTAL_IN ))
  if (( COVER_PCT >= COVER_SKIP_PCT )); then
    echo ">> Enough coverage (${COVER_PCT}% >= ${COVER_SKIP_PCT}%) — skipping heavy R3."
  else
    echo ">> Rescue 3: denser features + looser matching (new DB, remap + safe-promotion)"
    tqrs "$SPARSE0_DIR/logs/R3_feature_extractor.log" "$TO_SEQ_MIN" \
      colmap feature_extractor \
        --database_path "$DB_R3" \
        --image_path    "$IMAGES_DIR" \
        "${READER_OPTS[@]}" \
        "${SIFT_GPU[@]}" \
        "${SIFT_SAFE_OPTS[@]}" \
        --SiftExtraction.max_num_features 18000 \
        --SiftExtraction.peak_threshold 0.009
    tqrs "$SPARSE0_DIR/logs/R3_matcher.log" "$TO_HEAVY_MIN" \
      colmap exhaustive_matcher \
        --database_path "$DB_R3" \
        "${MATCH_GPU[@]}" \
        --SiftMatching.cross_check 1 \
        "${MATCH_SAFE_OPTS[@]}" \
        --SiftMatching.max_ratio 0.90 \
        --SiftMatching.max_num_matches 25000
    tqrs "$SPARSE0_DIR/logs/R3_mapper.log" "$TO_MAPPER_MIN" \
      colmap mapper \
        --database_path "$DB_R3" \
        --image_path    "$IMAGES_DIR" \
        --output_path   "$SPARSE_DIR" \
        --Mapper.init_min_num_inliers 42 \
        --Mapper.init_min_tri_angle 4 \
        --Mapper.init_max_error 8 \
        --Mapper.abs_pose_min_num_inliers 16 \
        --Mapper.multiple_models 0 \
        --Mapper.max_num_models 1
    promote_model_safely
    mark_model_origin "$SPARSE0_DIR" "$DB_R3"
    REG_NOW=$(count_registered_images "$SPARSE0_DIR")
  fi
fi

# Mask-guided attach again (only if --masks used)
if (( REG_NOW < TOTAL_IN )) && [[ "$USE_MASKS" == "true" ]]; then
  echo ">> Mask-guided neighbor seed + attach (post R3)"
  coarse_mask_attach_if_possible
  REG_NOW=$(count_registered_images "$SPARSE0_DIR")
fi

# ---------- Optional final BA tweak ----------
if [[ "$UNKNOWN_INTR" == "1" && "$SHARED_INTR" == "true" ]]; then
  echo ">> Final polish BA (refine principal point on shared intrinsics)"
  ensure_sparse_logs
  qrs "$SPARSE0_DIR/logs/99_polish_bundle_adjuster.log" \
    colmap bundle_adjuster \
      --input_path  "$SPARSE0_DIR" \
      --output_path "$SPARSE0_DIR" \
      --BundleAdjustment.refine_extrinsics 1 \
      --BundleAdjustment.refine_focal_length $BA_REFINE_F \
      --BundleAdjustment.refine_principal_point 1 \
      --BundleAdjustment.refine_extra_params $BA_REFINE_EXTRA
  colmap model_converter --input_path "$SPARSE0_DIR" --output_path "$SPARSE0_DIR" --output_type TXT >/dev/null 2>&1 || true
fi

# ============================================================
# Global final BA
# ============================================================
if [[ -d "$SPARSE0_DIR" ]]; then
  echo ">> Global final BA (safety pass before any undistortion/MVS)"
  ensure_sparse_logs
  qrs "$SPARSE0_DIR/logs/100_global_final_bundle_adjuster.log" \
    colmap bundle_adjuster \
      --input_path  "$SPARSE0_DIR" \
      --output_path "$SPARSE0_DIR" \
      --BundleAdjustment.refine_extrinsics 1 \
      --BundleAdjustment.refine_focal_length $BA_REFINE_F \
      --BundleAdjustment.refine_principal_point 0 \
      --BundleAdjustment.refine_extra_params $BA_REFINE_EXTRA
  colmap model_converter --input_path "$SPARSE0_DIR" --output_path "$SPARSE0_DIR" --output_type TXT >/dev/null 2>&1 || true
  colmap model_converter --input_path "$SPARSE0_DIR" --output_path "$SPARSE0_DIR/sparse.ply" --output_type PLY >/dev/null 2>&1 || true
fi

if [[ -d "$SPARSE0_DIR" ]]; then
  print_sparse_summary
fi

# ============================================================
# Skip sparse densification unless explicitly requested
# ============================================================
if [[ "${SPARSE_DENSIFY:-0}" == "1" || "$RUN_MVS_ULTRA" == "true" ]]; then
  echo ">> Optional point triangulation (enabled)"
  ensure_sparse_logs
  qrs "$SPARSE0_DIR/logs/04_point_triangulator.log" \
    colmap point_triangulator \
      --database_path "$(db_for_current_model)" \
      --image_path    "$IMAGES_DIR" \
      --input_path    "$SPARSE0_DIR" \
      --output_path   "$SPARSE0_DIR"
else
  echo ">> Skipping point_triangulator (no MVS and SPARSE_DENSIFY!=1)"
fi

# ============================================================
# Optional undistorted SfM export + mask undistortion
# ============================================================
if [[ "$UNDISTORT_IMAGES" == "true" ]]; then
  echo ">> Undistorting SfM (OPENCV -> PINHOLE) into UNDIST_DIR"
  mkdir -p "$UNDIST_DIR"
  ensure_sparse_logs
  qrs "$SPARSE0_DIR/logs/08_image_undistorter.log" \
    colmap image_undistorter \
      --image_path  "$IMAGES_DIR" \
      --input_path  "$SPARSE0_DIR" \
      --output_path "$UNDIST_DIR" \
      --output_type COLMAP \
      --min_scale 1.0

  mkdir -p "$UNDIST_DIR/sparse/logs"
  qrs "$UNDIST_DIR/sparse/logs/09_undist_model_to_txt.log" \
    colmap model_converter --input_path "$UNDIST_DIR/sparse" --output_path "$UNDIST_DIR/sparse" --output_type TXT
  qrs "$UNDIST_DIR/sparse/logs/10_undist_model_to_ply.log" \
    colmap model_converter --input_path "$UNDIST_DIR/sparse" --output_path "$UNDIST_DIR/sparse/sparse.ply" --output_type PLY

  ensure_sparse_zero_layout "$UNDIST_DIR/sparse"

  # Undistort masks if either:
  #  (a) --masks was used (USE_MASKS=true), or
  #  (b) masks/ exists and has files (even without --masks)
  SHOULD_UD_MASKS="false"
  if [[ "$USE_MASKS" == "true" ]]; then
    SHOULD_UD_MASKS="true"
  elif [[ -d "$MASKS_DIR" ]]; then
    shopt -s nullglob nocaseglob
    _mask_probe=( "$MASKS_DIR"/*.jpg "$MASKS_DIR"/*.jpeg "$MASKS_DIR"/*.png "$MASKS_DIR"/*.bmp "$MASKS_DIR"/*.tif "$MASKS_DIR"/*.tiff "$MASKS_DIR"/*.pgm "$MASKS_DIR"/*.ppm )
    shopt -u nullglob nocaseglob
    if (( ${#_mask_probe[@]} > 0 )); then
      SHOULD_UD_MASKS="true"
      echo ">> Found masks directory with files; undistorting masks into undist/masks (no SfM masking applied)"
    fi
  fi

  if [[ "$SHOULD_UD_MASKS" == "true" ]]; then
    echo ">> Undistorting masks (registered-only) into undist/masks"
    TMP_MASK_UD="$UNDIST_DIR/_mask_undist_tmp"
    mkdir -p "$TMP_MASK_UD"
    qrs "$UNDIST_DIR/sparse/logs/08b_mask_undistorter.log" \
      colmap image_undistorter \
        --image_path  "$MASKS_DIR" \
        --input_path  "$SPARSE0_DIR" \
        --output_path "$TMP_MASK_UD" \
        --output_type COLMAP \
        --min_scale 1.0

    if [[ -d "$TMP_MASK_UD/images" ]]; then
      mkdir -p "$UNDIST_DIR/masks"
      shopt -s nullglob
      count_copied=0
      for f in "$TMP_MASK_UD"/images/*; do
        base="$(basename "$f")"
        if [[ -f "$UNDIST_DIR/images/$base" ]]; then
          cp -f "$f" "$UNDIST_DIR/masks/$base"
          ((count_copied++)) || true
        fi
      done
      shopt -u nullglob
      echo ">> Undistorted mask files copied: ${count_copied}"

      if command -v mogrify >/dev/null 2>&1; then
        echo ">> Thresholding undistorted masks to binary (50%)"
        mogrify -colorspace Gray -threshold 50% "$UNDIST_DIR/masks/"*
      else
        echo ">> mogrify not found; leaving undistorted masks as-is"
      fi
    else
      echo ">> Mask undistorter produced no images at: $TMP_MASK_UD/images"
    fi
    rm -rf "$TMP_MASK_UD" 2>/dev/null || true
  fi

  # Count poses in undistorted model for visibility
  if [[ -f "$UNDIST_DIR/sparse/images.txt" ]]; then
    und_reg="$(count_registered_images "$UNDIST_DIR/sparse")"
  elif [[ -f "$UNDIST_DIR/sparse/0/images.txt" ]]; then
    und_reg="$(count_registered_images "$UNDIST_DIR/sparse/0")"
  else
    und_reg=0
  fi
  und_total="$(count_all_input_images)"
  und_pct=$(( und_total>0 ? (100 * und_reg / und_total) : 0 ))
  echo ">> Undistorted summary: poses captured = ${und_reg} / ${und_total} (${und_pct}%)"
fi

# ============================================================
# MVS (reuses UNDIST if provided; robust stereo cfgs)
# ============================================================
if [[ "$RUN_MVS_ULTRA" == "true" ]]; then
  echo ">> Running MVS"
  rm -rf "$MVS_DIR"
  mkdir -p "$MVS_DIR/logs"

  if [[ "$UNDISTORT_IMAGES" == "true" && -d "$UNDIST_DIR" ]]; then
    echo ">> MVS input: reusing UNDIST_DIR (no re-undistortion)"
    [[ -d "$UNDIST_DIR/images" ]] || { echo "!! Missing $UNDIST_DIR/images"; exit 2; }
    [[ -d "$UNDIST_DIR/sparse"  ]] || { echo "!! Missing $UNDIST_DIR/sparse";  exit 2; }

    rm -rf "$MVS_DIR/images" 2>/dev/null || true
    ln -s "../undist/images" "$MVS_DIR/images"

    if [[ -d "$UNDIST_DIR/sparse" && ! -d "$UNDIST_DIR/sparse/0" ]]; then
      mkdir -p "$UNDIST_DIR/sparse/0"; shopt -s nullglob
      for f in cameras.bin images.bin points3D.bin cameras.txt images.txt points3D.txt; do
        if [[ -f "$UNDIST_DIR/sparse/$f" ]]; then
          mv "$UNDIST_DIR/sparse/$f" "$UNDIST_DIR/sparse/0/$f"
          ln -s "0/$f" "$UNDIST_DIR/sparse/$f"
        fi
      done
      shopt -u nullglob
    fi
    mkdir -p "$MVS_DIR/sparse"
    rm -rf "$MVS_DIR/sparse/0" 2>/dev/null || true
    ln -s "../../undist/sparse/0" "$MVS_DIR/sparse/0"
    shopt -s nullglob
    for f in cameras.bin images.bin points3D.bin cameras.txt images.txt points3D.txt; do
      [[ -f "$UNDIST_DIR/sparse/0/$f" ]] && { rm -f "$MVS_DIR/sparse/$f"; ln -s "0/$f" "$MVS_DIR/sparse/$f"; }
    done
    shopt -u nullglob

    rm -rf "$MVS_DIR/stereo" 2>/dev/null || true
    if [[ -d "$UNDIST_DIR/stereo" ]]; then
      ln -s "../undist/stereo" "$MVS_DIR/stereo"
    else
      echo ">> UNDIST_DIR/stereo missing; writing minimal stereo cfgs"
      mkdir -p "$MVS_DIR/stereo"
      cat > "$MVS_DIR/stereo/patch-match.cfg" <<'CFG'
PatchMatchStereo.max_image_size = -1
PatchMatchStereo.window_radius = 5
PatchMatchStereo.num_iterations = 5
PatchMatchStereo.geom_consistency = 0
PatchMatchStereo.filter = 0
CFG
      cat > "$MVS_DIR/stereo/fusion.cfg" <<'CFG'
StereoFusion.max_image_size = -1
StereoFusion.min_num_pixels = 5
StereoFusion.max_reproj_error = 2.0
StereoFusion.max_depth_error = 0.03
StereoFusion.max_normal_error = 25
CFG
    fi

    qr "$MVS_DIR/logs/01_mvs_model_to_txt.log" \
      colmap model_converter --input_path "$MVS_DIR/sparse/0" --output_path "$MVS_DIR/sparse/0" --output_type TXT

  else
    echo ">> MVS input: undistorting from SPARSE0_DIR into MVS_DIR"
    qr "$MVS_DIR/logs/00_image_undistorter.log" \
      colmap image_undistorter \
        --image_path  "$IMAGES_DIR" \
        --input_path  "$SPARSE0_DIR" \
        --output_path "$MVS_DIR" \
        --output_type COLMAP \
        --min_scale 1.0

    if [[ -d "$MVS_DIR/sparse" && ! -d "$MVS_DIR/sparse/0" ]]; then
      mkdir -p "$MVS_DIR/sparse/0"; shopt -s nullglob
      for f in cameras.bin images.bin points3D.bin cameras.txt images.txt points3D.txt; do
        [[ -f "$MVS_DIR/sparse/$f" ]] && { mv "$MVS_DIR/sparse/$f" "$MVS_DIR/sparse/0/$f"; ln -s "0/$f" "$MVS_DIR/sparse/$f"; }
      done
      shopt -u nullglob
    fi

    if [[ ! -d "$MVS_DIR/stereo" ]]; then
      echo ">> Creating minimal stereo cfgs in MVS_DIR"
      mkdir -p "$MVS_DIR/stereo"
      cat > "$MVS_DIR/stereo/patch-match.cfg" <<'CFG'
PatchMatchStereo.max_image_size = -1
PatchMatchStereo.window_radius = 5
PatchMatchStereo.num_iterations = 5
PatchMatchStereo.geom_consistency = 0
PatchMatchStereo.filter = 0
CFG
      cat > "$MVS_DIR/stereo/fusion.cfg" <<'CFG'
StereoFusion.max_image_size = -1
StereoFusion.min_num_pixels = 5
StereoFusion.max_reproj_error = 2.0
StereoFusion.max_depth_error = 0.03
StereoFusion.max_normal_error = 25
CFG
    fi

    qr "$MVS_DIR/logs/01_mvs_model_to_txt.log" \
      colmap model_converter --input_path "$MVS_DIR/sparse/0" --output_path "$MVS_DIR/sparse/0" --output_type TXT
  fi

  echo ">> MVS backend: COLMAP PatchMatch (GPU → CPU fallback if needed)"
  run_pms_try() {
    local tag="$1" maxsz="$2" wr="$3" it="$4" gpu="$5"
    local log="$MVS_DIR/logs/11_patch_match_stereo_${tag}.log"
    local gpu_flag=( ); if [[ "$gpu" == "gpu" ]]; then gpu_flag=( --PatchMatchStereo.gpu_index 0 ); else gpu_flag=( --PatchMatchStereo.gpu_index -1 ); fi
    if ! ( colmap patch_match_stereo \
            --workspace_path "$MVS_DIR" \
            --workspace_format COLMAP \
            "${gpu_flag[@]}" \
            --PatchMatchStereo.max_image_size "$maxsz" \
            --PatchMatchStereo.window_radius "$wr" \
            --PatchMatchStereo.num_iterations "$it" \
            --PatchMatchStereo.geom_consistency 0 \
            --PatchMatchStereo.filter 0 > "$log" 2>&1 ); then
      echo "!! PatchMatch (${tag}) failed."
      { echo "----- tail of $log -----"; tail -n 120 "$log" || true; echo "--------------------------"; } >&2
      if grep -qE 'No CUDA devices available|num_cuda_devices' "$log"; then return 42; fi
      return 1
    fi
    return 0
  }

  if ! run_pms_try "full"  -1   7 12 "gpu"; then
    if [[ $? -eq 42 ]]; then
      echo ">> GPU PatchMatch unavailable — switching to CPU PatchMatch"
      run_pms_try "cpu_3072" 3072 6 10 "cpu" || run_pms_try "cpu_2560" 2560 5 8 "cpu" || true
    else
      run_pms_try "cpu_3072" 3072 6 10 "cpu" || true
    fi
  fi

  DEPTH_DIR="$MVS_DIR/stereo/depth_maps"
  shopt -s nullglob
  PHOTOS=( "$DEPTH_DIR"/*.photometric.bin ); GEOMS=( "$DEPTH_DIR"/*.geometric.bin ); shopt -u nullglob
  INPUT_TYPE=""
  if (( ${#PHOTOS[@]} )); then INPUT_TYPE="photometric"; echo ">> Depth maps found: photometric (${#PHOTOS[@]})"
  elif (( ${#GEOMS[@]} )); then INPUT_TYPE="geometric"; echo ">> Depth maps found: geometric (${#GEOMS[@]})"
  else echo "!! PatchMatch produced no depth maps — cannot fuse. See logs in: $MVS_DIR/logs"; exit 2; fi

  run_fusion_try() {
    local tag="$1" itype="$2" minpix="$3" reproj="$4" deptherr="$5" normalerr="$6" cache="$7"
    local log="$MVS_DIR/logs/12_stereo_fusion_${tag}.log"
    if ! ( colmap stereo_fusion \
            --workspace_path "$MVS_DIR" \
            --workspace_format COLMAP \
            --input_type "$itype" \
            --output_path "$MVS_DIR/fused.ply" \
            --StereoFusion.max_image_size -1 \
            --StereoFusion.min_num_pixels "$minpix" \
            --StereoFusion.max_reproj_error "$reproj" \
            --StereoFusion.max_depth_error "$deptherr" \
            --StereoFusion.max_normal_error "$normalerr" \
            --StereoFusion.cache_size "$cache" > "$log" 2>&1 ); then
      echo "!! StereoFusion (${tag}) failed."
      { echo "----- tail of $log -----"; tail -n 120 "$log" || true; echo "--------------------------"; } >&2
      return 1
    fi
    return 0
  }

  if [[ "$INPUT_TYPE" == "photometric" ]]; then
    run_fusion_try "P_strict" photometric 10 1.0 0.010 15 24 \
      || run_fusion_try "P_relax"  photometric 8  1.5 0.020 20 24 \
      || run_fusion_try "P_loose"  photometric 5  2.0 0.030 25 24 || true
  else
    run_fusion_try "G_strict" geometric 10 1.0 0.010 7.5 24 \
      || run_fusion_try "G_relax"  geometric 8  1.5 0.020 12.5 24 \
      || run_fusion_try "G_loose"  geometric 5  2.0 0.030 20.0 24 || true
  fi

  if [[ -s "$MVS_DIR/fused.ply" ]]; then
    echo "Dense cloud: $MVS_DIR/fused.ply"
    mkdir -p "$MVS_DIR/sparse/0" 2>/dev/null || true
    ln -sfn "$MVS_DIR/fused.ply" "$MVS_DIR/sparse/0/points3D.ply"
  else
    echo "!! No dense point cloud produced. Logs: $MVS_DIR/logs"
  fi
fi


# ============================================================
# Final cleanup of auxiliary files and folders
# ============================================================
cleanup_auxiliary_artifacts() {
  # 1) Intermediate COLMAP databases from the different passes
  rm -f "$DB_P1" "$DB_R2" "$DB_R3" 2>/dev/null || true

  # 2) Temporary log directory
  rm -rf "$TMP_LOGS" 2>/dev/null || true

  # 3) Log folders
  if [[ "${KEEP_COLMAP_LOGS:-0}" != "1" ]]; then
    rm -rf "$SPARSE0_DIR/logs" 2>/dev/null || true
    rm -rf "$UNDIST_DIR/sparse/logs" 2>/dev/null || true
    rm -rf "$MVS_DIR/logs" 2>/dev/null || true
  fi

  # 4) COLMAP helper scripts in UNDIST_DIR
  rm -f "$UNDIST_DIR/run-colmap-photometric.sh" \
        "$UNDIST_DIR/run-colmap-geometric.sh" 2>/dev/null || true

  # 5) Stereo workspaces:
  if [[ "${KEEP_UNDIST_STEREO:-0}" != "1" ]]; then
    rm -rf "$UNDIST_DIR/stereo" 2>/dev/null || true
  fi

  if [[ "${KEEP_MVS_STEREO:-0}" != "1" ]]; then
    rm -rf "$MVS_DIR/stereo" 2>/dev/null || true
  fi
}

cleanup_auxiliary_artifacts
