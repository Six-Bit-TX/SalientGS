#!/bin/bash
# =============================================================================
# GLOMAP + sgs-joint ETH3D SLAM Evaluation
#
# Pipeline per scene:
#   1. COLMAP SIFT feature extraction
#   2. COLMAP sequential matching
#   3. GLOMAP global mapper
#   4. Pose evaluation (SfM-only)
#   5. sgs-joint GS optimization
#   6. Pose evaluation (joint-optimized)
#
# Runs scenes in parallel across all available GPUs.
# Uses a separate work directory to avoid interfering with FastMap results.
#
# Usage:
#   bash scripts/run_glomap_eth3d_slam.sh
# =============================================================================

export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_HOME="${CONDA_PREFIX:-/data16t/xty/miniconda3/envs/gsplat}"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CPATH"
export PYTHONHTTPSVERIFY=0
export CURL_CA_BUNDLE=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

ETH3D_ZIP_DIR="$WORKSPACE_ROOT/ETH3D_SLAM"
SOURCE_DIR="$WORKSPACE_ROOT/ETH3D_SLAM_workdir"
GLOMAP_DIR="$WORKSPACE_ROOT/ETH3D_SLAM_glomap"
RESULTS_DIR="$WORKSPACE_ROOT/eth3d_slam_glomap_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="$RESULTS_DIR/glomap_eth3d_slam_$TIMESTAMP.csv"

COLMAP_BIN="/usr/local/bin/colmap"
GLOMAP_BIN="$(which glomap 2>/dev/null || echo "glomap")"

NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected."
    exit 1
fi
echo "Detected $NUM_GPUS GPUs."

GPU_LOCK_DIR="/tmp/glomap_eth3d_gpu_locks"
SCENE_PREFIXES=(cables camera ceiling desk einstein kidnap large mannequin motion planar plant reflective repetitive sfm sofa table vicon)

mkdir -p "$GLOMAP_DIR" "$RESULTS_DIR" "$GPU_LOCK_DIR"

echo "config,scene,prefix,num_images,num_registered,num_matched,recall_0.1m,auc_0.1m,auc_0.5m,median_error_m,sfm_time_s,joint_time_s,total_time_s" > "$CSV_FILE"

# ---- GPU lock helpers ----
acquire_gpu() {
    while true; do
        for gpu in $(seq 0 $((NUM_GPUS - 1))); do
            lockfile="$GPU_LOCK_DIR/gpu_${gpu}.lock"
            if ( set -o noclobber; echo $$ > "$lockfile" ) 2>/dev/null; then
                echo "$gpu"
                return 0
            fi
        done
        sleep 5
    done
}

release_gpu() {
    local gpu="$1"
    rm -f "$GPU_LOCK_DIR/gpu_${gpu}.lock"
}

rm -f "$GPU_LOCK_DIR"/gpu_*.lock

get_prefix() {
    local scene="$1"
    for prefix in "${SCENE_PREFIXES[@]}"; do
        if [[ "$scene" == ${prefix}* ]]; then
            echo "$prefix"
            return
        fi
    done
    echo "$scene"
}

# =============================================================================
# STEP 1: Ensure source scenes are extracted
# =============================================================================
extract_scenes() {
    echo ""
    echo "================================================================"
    echo "STEP 1: Verifying ETH3D SLAM scene data"
    echo "================================================================"

    local count=0

    # Extract from zips if source dir doesn't have them
    if [ -d "$ETH3D_ZIP_DIR" ]; then
        for zipfile in "$ETH3D_ZIP_DIR"/*_mono.zip; do
            [ -f "$zipfile" ] || continue
            local scene_name=$(basename "$zipfile" _mono.zip)
            local scene_dir="$SOURCE_DIR/$scene_name"

            if [ -d "$scene_dir/rgb" ]; then
                : # already extracted
            else
                echo "  Extracting $scene_name ..."
                mkdir -p "$SOURCE_DIR"
                unzip -q -o "$zipfile" -d "$SOURCE_DIR/"
            fi

            if [ ! -e "$scene_dir/images" ]; then
                ln -sf "$scene_dir/rgb" "$scene_dir/images"
            fi
            count=$((count + 1))
        done
    fi

    echo "  $count scenes available."
}

# =============================================================================
# STEP 2: Run GLOMAP pipeline on a single scene
# =============================================================================
run_scene() {
    local scene_name="$1"
    local gpu="$2"
    local source_scene="$SOURCE_DIR/$scene_name"
    local scene_dir="$GLOMAP_DIR/$scene_name"
    local prefix=$(get_prefix "$scene_name")

    local result_dir="$RESULTS_DIR/$scene_name"
    local log_file="$result_dir/run.log"
    mkdir -p "$result_dir"

    if [ -f "$result_dir/done.marker" ]; then
        echo "[GPU $gpu] $scene_name: already done, skipping."
        return 0
    fi

    if [ ! -d "$source_scene/rgb" ]; then
        echo "[GPU $gpu] $scene_name: no rgb/ directory in source, skipping."
        return 1
    fi

    if [ ! -f "$source_scene/groundtruth.txt" ]; then
        echo "[GPU $gpu] $scene_name: no groundtruth.txt in source, skipping."
        return 1
    fi

    echo "[GPU $gpu] $scene_name (prefix=$prefix): starting GLOMAP pipeline..."

    # ---- Setup scene directory with symlinks to source data ----
    mkdir -p "$scene_dir"
    ln -sf "$source_scene/rgb" "$scene_dir/rgb" 2>/dev/null
    if [ -L "$source_scene/images" ] || [ -d "$source_scene/images" ]; then
        ln -sf "$source_scene/images" "$scene_dir/images" 2>/dev/null
    else
        ln -sf "$scene_dir/rgb" "$scene_dir/images" 2>/dev/null
    fi
    for f in groundtruth.txt rgb.txt calibration.txt config.yaml; do
        [ -f "$source_scene/$f" ] && ln -sf "$source_scene/$f" "$scene_dir/$f" 2>/dev/null
    done

    # Clean previous GLOMAP artifacts
    rm -f "$scene_dir/database.db"
    rm -rf "$scene_dir/sparse" "$scene_dir/gsplat"

    local total_start=$(date +%s.%N)
    local sfm_start=$(date +%s.%N)

    # ---- Stage 1: COLMAP Feature Extraction ----
    {
        echo "=== Stage 1: COLMAP Feature Extraction ==="
        CUDA_VISIBLE_DEVICES=$gpu $COLMAP_BIN feature_extractor \
            --database_path "$scene_dir/database.db" \
            --image_path "$scene_dir/images" \
            --ImageReader.camera_model SIMPLE_RADIAL \
            --ImageReader.single_camera 1 \
            --SiftExtraction.num_threads -1 \
            --SiftExtraction.max_num_features 8192 \
            --SiftExtraction.max_image_size 3200 \
            --SiftExtraction.use_gpu 1 \
            --SiftExtraction.gpu_index 0 \
            2>&1
    } > "$log_file" 2>&1

    # ---- Stage 2: COLMAP Sequential Matching ----
    {
        echo ""
        echo "=== Stage 2: COLMAP Sequential Matching ==="
        CUDA_VISIBLE_DEVICES=$gpu $COLMAP_BIN sequential_matcher \
            --database_path "$scene_dir/database.db" \
            --SiftMatching.use_gpu 1 \
            --SiftMatching.gpu_index 0 \
            --SiftMatching.guided_matching 1 \
            --SiftMatching.max_num_matches 32768 \
            --SequentialMatching.overlap 10 \
            --SequentialMatching.quadratic_overlap 1 \
            --SequentialMatching.loop_detection 0 \
            2>&1
    } >> "$log_file" 2>&1

    # ---- Stage 3: GLOMAP Mapper ----
    {
        echo ""
        echo "=== Stage 3: GLOMAP Mapper ==="
        mkdir -p "$scene_dir/sparse"
        # --skip_retriangulation 1: workaround for COLMAP/GLOMAP ABI mismatch
        # in RetriangulateTracks (frame.DataIds() check failure).
        CUDA_VISIBLE_DEVICES=$gpu $GLOMAP_BIN mapper \
            --database_path "$scene_dir/database.db" \
            --image_path "$scene_dir/images" \
            --output_path "$scene_dir/sparse" \
            --skip_retriangulation 1 \
            2>&1
    } >> "$log_file" 2>&1

    local sfm_end=$(date +%s.%N)
    local sfm_time=$(echo "$sfm_end - $sfm_start" | bc)

    # ---- Evaluate SfM-only poses ----
    local sfm_eval_json="$result_dir/sfm_eval.json"
    {
        echo ""
        echo "=== SfM Pose Evaluation ==="
        python "$SCRIPT_DIR/eval_eth3d_slam_poses.py" \
            --scene_dir "$scene_dir" \
            --sparse_dir "$scene_dir/sparse/0" \
            --output "$sfm_eval_json" 2>&1
    } >> "$log_file" 2>&1

    local sfm_recall sfm_auc01 sfm_auc05 sfm_nimgs sfm_nreg sfm_nmatched sfm_median
    sfm_recall=$(python3 -c "import json; m=list(json.load(open('$sfm_eval_json'))['per_scene'].values())[0]; print(m.get('recall_0.1m',0))" 2>/dev/null || echo "0")
    sfm_auc01=$(python3 -c "import json; m=list(json.load(open('$sfm_eval_json'))['per_scene'].values())[0]; print(m.get('auc_0.1m',0))" 2>/dev/null || echo "0")
    sfm_auc05=$(python3 -c "import json; m=list(json.load(open('$sfm_eval_json'))['per_scene'].values())[0]; print(m.get('auc_0.5m',0))" 2>/dev/null || echo "0")
    sfm_nimgs=$(python3 -c "import json; m=list(json.load(open('$sfm_eval_json'))['per_scene'].values())[0]; print(m.get('num_images',0))" 2>/dev/null || echo "0")
    sfm_nreg=$(python3 -c "import json; m=list(json.load(open('$sfm_eval_json'))['per_scene'].values())[0]; print(m.get('num_registered',0))" 2>/dev/null || echo "0")
    sfm_nmatched=$(python3 -c "import json; m=list(json.load(open('$sfm_eval_json'))['per_scene'].values())[0]; print(m.get('num_matched',0))" 2>/dev/null || echo "0")
    sfm_median=$(python3 -c "import json; m=list(json.load(open('$sfm_eval_json'))['per_scene'].values())[0]; print(m.get('median_error','inf'))" 2>/dev/null || echo "inf")

    echo "sfm_only,$scene_name,$prefix,$sfm_nimgs,$sfm_nreg,$sfm_nmatched,$sfm_recall,$sfm_auc01,$sfm_auc05,$sfm_median,$sfm_time,0,$sfm_time" >> "$CSV_FILE"

    # ---- Stage 4: Joint GS Optimization ----
    local joint_start=$(date +%s.%N)

    # Only run sgs-joint if GLOMAP produced a reconstruction
    if [ -d "$scene_dir/sparse/0" ] && [ -n "$(ls -A "$scene_dir/sparse/0/" 2>/dev/null)" ]; then
        {
            echo ""
            echo "=== Stage 4: Joint GS Optimization ==="
            WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$gpu \
                python "$PROJECT_ROOT/salientgs/vis/gsplat_joint.py" mcmc_importance \
                --data_dir "$scene_dir" \
                --image_folder_name "images" \
                --data_factor 1 \
                --result_dir "$scene_dir/gsplat" \
                --no-use_wandb \
                --test_every 0 \
                --strategy.cap-max 1500000 \
                2>&1
        } >> "$log_file" 2>&1
    else
        echo "  Skipping sgs-joint: no GLOMAP reconstruction." >> "$log_file"
    fi

    local joint_end=$(date +%s.%N)
    local joint_time=$(echo "$joint_end - $joint_start" | bc)
    local total_end=$(date +%s.%N)
    local total_time=$(echo "$total_end - $total_start" | bc)

    # ---- Evaluate joint-optimized poses ----
    local joint_eval_json="$result_dir/joint_eval.json"
    local ckpt_dir="$scene_dir/gsplat/ckpts"
    local latest_ckpt=""
    if [ -d "$ckpt_dir" ]; then
        latest_ckpt=$(ls -1 "$ckpt_dir"/ckpt_*_rank0.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
    fi

    {
        echo ""
        echo "=== Joint Pose Evaluation ==="
        if [ -n "$latest_ckpt" ]; then
            echo "Using checkpoint: $latest_ckpt"
            python "$SCRIPT_DIR/eval_eth3d_slam_poses.py" \
                --scene_dir "$scene_dir" \
                --sparse_dir "$scene_dir/sparse/0" \
                --ckpt "$latest_ckpt" \
                --output "$joint_eval_json" 2>&1
        else
            echo "No checkpoint found, copying SfM eval as joint eval."
            cp "$sfm_eval_json" "$joint_eval_json" 2>/dev/null
        fi
    } >> "$log_file" 2>&1

    local joint_recall joint_auc01 joint_auc05 joint_nreg joint_nmatched joint_median
    joint_recall=$(python3 -c "import json; m=list(json.load(open('$joint_eval_json'))['per_scene'].values())[0]; print(m.get('recall_0.1m',0))" 2>/dev/null || echo "0")
    joint_auc01=$(python3 -c "import json; m=list(json.load(open('$joint_eval_json'))['per_scene'].values())[0]; print(m.get('auc_0.1m',0))" 2>/dev/null || echo "0")
    joint_auc05=$(python3 -c "import json; m=list(json.load(open('$joint_eval_json'))['per_scene'].values())[0]; print(m.get('auc_0.5m',0))" 2>/dev/null || echo "0")
    joint_nreg=$(python3 -c "import json; m=list(json.load(open('$joint_eval_json'))['per_scene'].values())[0]; print(m.get('num_registered',0))" 2>/dev/null || echo "0")
    joint_nmatched=$(python3 -c "import json; m=list(json.load(open('$joint_eval_json'))['per_scene'].values())[0]; print(m.get('num_matched',0))" 2>/dev/null || echo "0")
    joint_median=$(python3 -c "import json; m=list(json.load(open('$joint_eval_json'))['per_scene'].values())[0]; print(m.get('median_error','inf'))" 2>/dev/null || echo "inf")

    echo "joint,$scene_name,$prefix,$sfm_nimgs,$joint_nreg,$joint_nmatched,$joint_recall,$joint_auc01,$joint_auc05,$joint_median,$sfm_time,$joint_time,$total_time" >> "$CSV_FILE"

    touch "$result_dir/done.marker"
    echo "[GPU $gpu] $scene_name: SfM(R@0.1=${sfm_recall}% reg=${sfm_nreg}) Joint(R@0.1=${joint_recall}%) time=${total_time}s"
}

# ---- Job wrapper with GPU locking ----
run_job() {
    local scene_name="$1"
    local gpu=$(acquire_gpu)
    run_scene "$scene_name" "$gpu"
    release_gpu "$gpu"
}

# =============================================================================
# STEP 3: Run all scenes in parallel
# =============================================================================
run_all_scenes() {
    echo ""
    echo "================================================================"
    echo "STEP 2: Running GLOMAP pipeline on all scenes"
    echo "================================================================"

    local all_scenes=()
    for scene_dir in "$SOURCE_DIR"/*/; do
        [ -d "$scene_dir/rgb" ] || continue
        [ -f "$scene_dir/groundtruth.txt" ] || continue
        local scene_name=$(basename "$scene_dir")
        all_scenes+=("$scene_name")
    done

    echo "Found ${#all_scenes[@]} scenes to process."
    echo ""

    for scene_name in "${all_scenes[@]}"; do
        run_job "$scene_name" &
    done
    wait

    echo ""
    echo "All scenes processed."
}

# =============================================================================
# STEP 4: Generate summary with GLOMAP paper reference
# =============================================================================
generate_summary() {
    echo ""
    echo "================================================================"
    echo "STEP 3: Generating Summary"
    echo "================================================================"

    # Batch evaluation
    local sfm_summary="$RESULTS_DIR/glomap_sfm_summary.json"
    python "$SCRIPT_DIR/eval_eth3d_slam_poses.py" \
        --batch_dir "$GLOMAP_DIR" \
        --sparse_subdir "sparse/0" \
        --output "$sfm_summary" 2>/dev/null || true

    local joint_summary="$RESULTS_DIR/glomap_joint_summary.json"
    python "$SCRIPT_DIR/eval_eth3d_slam_poses.py" \
        --batch_dir "$GLOMAP_DIR" \
        --sparse_subdir "sparse/0" \
        --ckpt_subdir "gsplat/ckpts" \
        --output "$joint_summary" 2>/dev/null || true

    python3 - "$sfm_summary" "$joint_summary" << 'PYEOF'
import json, sys

prefixes = ['cables','camera','ceiling','desk','einstein','kidnap','large',
            'mannequin','motion','planar','plant','reflective','repetitive',
            'sfm','sofa','table','vicon']

# GLOMAP paper Table 1 reference (from arxiv.tex)
paper_ref = {
    'cables':     {'recall': 88.0, 'auc01': 76.2, 'auc05': 85.6},
    'camera':     {'recall': 32.6, 'auc01': 21.9, 'auc05': 30.8},
    'ceiling':    {'recall': 28.7, 'auc01': 22.3, 'auc05': 27.4},
    'desk':       {'recall': 32.3, 'auc01': 28.5, 'auc05': 31.6},
    'einstein':   {'recall': 48.5, 'auc01': 36.5, 'auc05': 47.9},
    'kidnap':     {'recall': 73.3, 'auc01': 70.3, 'auc05': 72.7},
    'large':      {'recall': 49.0, 'auc01': 45.8, 'auc05': 48.4},
    'mannequin':  {'recall': 67.4, 'auc01': 61.4, 'auc05': 66.4},
    'motion':     {'recall': 39.8, 'auc01': 22.5, 'auc05': 45.9},
    'planar':     {'recall':100.0, 'auc01': 98.7, 'auc05': 99.7},
    'plant':      {'recall': 93.3, 'auc01': 82.0, 'auc05': 93.4},
    'reflective': {'recall': 22.0, 'auc01': 12.1, 'auc05': 31.3},
    'repetitive': {'recall': 32.7, 'auc01': 29.2, 'auc05': 32.0},
    'sfm':        {'recall': 97.0, 'auc01': 79.6, 'auc05': 95.2},
    'sofa':       {'recall': 23.9, 'auc01': 22.1, 'auc05': 23.5},
    'table':      {'recall': 94.3, 'auc01': 84.3, 'auc05': 92.3},
    'vicon':      {'recall': 97.0, 'auc01': 80.5, 'auc05': 93.7},
    'average':    {'recall': 66.4, 'auc01': 57.0, 'auc05': 65.7},
}

def load_summary(path):
    try:
        with open(path) as f:
            return json.load(f).get('summary', {})
    except Exception:
        return {}

sfm = load_summary(sys.argv[1])
joint = load_summary(sys.argv[2])

hdr = (f"{'Prefix':<13} "
       f"{'R@0.1m':>7} {'A@0.1m':>7} {'A@0.5m':>7}  "
       f"{'R@0.1m':>7} {'A@0.1m':>7} {'A@0.5m':>7}  "
       f"{'R@0.1m':>7} {'A@0.1m':>7} {'A@0.5m':>7}")
sep = "=" * len(hdr)

print()
print(sep)
print("ETH3D SLAM: GLOMAP Evaluation (COLMAP benchmark protocol)")
print(sep)
print(f"{'':13} {'--- SfM Only ---':^23}  {'--- + sgs-joint ---':^23}  {'--- Paper (ref) ---':^23}")
print(hdr)
print("-" * len(hdr))

for p in prefixes:
    sm = sfm.get(p, {})
    sr = sm.get('recall_0.1m', 0)
    sa1 = sm.get('auc_0.1m', 0)
    sa5 = sm.get('auc_0.5m', 0)

    jm = joint.get(p, {})
    jr = jm.get('recall_0.1m', 0)
    ja1 = jm.get('auc_0.1m', 0)
    ja5 = jm.get('auc_0.5m', 0)

    ref = paper_ref.get(p, {})
    rr = ref.get('recall', 0)
    ra1 = ref.get('auc01', 0)
    ra5 = ref.get('auc05', 0)

    print(f"{p:<13} {sr:>7.1f} {sa1:>7.1f} {sa5:>7.1f}  {jr:>7.1f} {ja1:>7.1f} {ja5:>7.1f}  {rr:>7.1f} {ra1:>7.1f} {ra5:>7.1f}")

print("-" * len(hdr))
sm = sfm.get('average', {})
jm = joint.get('average', {})
ref = paper_ref['average']
print(f"{'Average':<13} "
      f"{sm.get('recall_0.1m',0):>7.1f} {sm.get('auc_0.1m',0):>7.1f} {sm.get('auc_0.5m',0):>7.1f}  "
      f"{jm.get('recall_0.1m',0):>7.1f} {jm.get('auc_0.1m',0):>7.1f} {jm.get('auc_0.5m',0):>7.1f}  "
      f"{ref['recall']:>7.1f} {ref['auc01']:>7.1f} {ref['auc05']:>7.1f}")
print(sep)
PYEOF

    echo ""
    echo "Results CSV: $CSV_FILE"
    echo "SfM summary: $sfm_summary"
    echo "Joint summary: $joint_summary"
}

# =============================================================================
# MAIN
# =============================================================================
echo "========================================"
echo "GLOMAP + sgs-joint ETH3D SLAM Evaluation"
echo "Date: $(date)"
echo "Source dir: $SOURCE_DIR"
echo "GLOMAP dir: $GLOMAP_DIR"
echo "Results: $RESULTS_DIR"
echo "CSV: $CSV_FILE"
echo "COLMAP: $COLMAP_BIN"
echo "GLOMAP: $GLOMAP_BIN"
echo "GPUs: $NUM_GPUS"
echo "========================================"

extract_scenes
run_all_scenes
generate_summary

echo ""
echo "========================================"
echo "GLOMAP ETH3D SLAM evaluation complete!"
echo "========================================"
