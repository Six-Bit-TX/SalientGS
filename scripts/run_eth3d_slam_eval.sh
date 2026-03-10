#!/bin/bash
# =============================================================================
# ETH3D SLAM Pose Evaluation Runner
#
# Extracts ETH3D SLAM mono datasets, runs the SalientGS pipeline
# (sgs-feat -> sgs-sfm -> sgs-joint), evaluates SfM-only and
# joint-optimized poses against ground truth, and collects results.
#
# Runs scenes in parallel across all available GPUs.
#
# Usage:
#   bash scripts/run_eth3d_slam_eval.sh
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
WORK_DIR="$WORKSPACE_ROOT/ETH3D_SLAM_workdir"
RESULTS_DIR="$WORKSPACE_ROOT/eth3d_slam_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="$RESULTS_DIR/eth3d_slam_$TIMESTAMP.csv"

NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected."
    exit 1
fi
echo "Detected $NUM_GPUS GPUs."

GPU_LOCK_DIR="/tmp/eth3d_slam_gpu_locks"

SCENE_PREFIXES=(cables camera ceiling desk einstein kidnap large mannequin motion planar plant reflective repetitive sfm sofa table vicon)

mkdir -p "$WORK_DIR" "$RESULTS_DIR" "$GPU_LOCK_DIR"

echo "config,scene,prefix,recall_0.1m,auc_0.1m,auc_0.5m,sfm_time_s,joint_time_s,total_time_s" > "$CSV_FILE"

# ---- GPU lock helpers (same pattern as run_ablation_study.sh) ----
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

# ---- Get scene prefix ----
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
# STEP 1: Extract all mono zip files
# =============================================================================
extract_scenes() {
    echo ""
    echo "================================================================"
    echo "STEP 1: Extracting ETH3D SLAM mono datasets"
    echo "================================================================"

    local count=0
    for zipfile in "$ETH3D_ZIP_DIR"/*_mono.zip; do
        [ -f "$zipfile" ] || continue
        local scene_name=$(basename "$zipfile" _mono.zip)
        local scene_dir="$WORK_DIR/$scene_name"

        if [ -d "$scene_dir/rgb" ]; then
            echo "  $scene_name: already extracted, skipping."
        else
            echo "  Extracting $scene_name ..."
            unzip -q -o "$zipfile" -d "$WORK_DIR/"
        fi

        # Create images/ symlink -> rgb/ (pipeline expects images/)
        if [ ! -e "$scene_dir/images" ]; then
            ln -s "$scene_dir/rgb" "$scene_dir/images"
        fi

        count=$((count + 1))
    done
    echo "  Extracted/verified $count scenes."
}

# =============================================================================
# STEP 2: Run pipeline on a single scene
# =============================================================================
run_scene() {
    local scene_name="$1"
    local gpu="$2"
    local scene_dir="$WORK_DIR/$scene_name"
    local prefix=$(get_prefix "$scene_name")

    local result_dir="$RESULTS_DIR/$scene_name"
    local log_file="$result_dir/run.log"
    mkdir -p "$result_dir"

    if [ -f "$result_dir/done.marker" ]; then
        echo "[GPU $gpu] $scene_name: already done, skipping."
        return 0
    fi

    if [ ! -d "$scene_dir/rgb" ]; then
        echo "[GPU $gpu] $scene_name: no rgb/ directory, skipping."
        return 1
    fi

    if [ ! -f "$scene_dir/groundtruth.txt" ]; then
        echo "[GPU $gpu] $scene_name: no groundtruth.txt, skipping."
        return 1
    fi

    echo "[GPU $gpu] $scene_name (prefix=$prefix): starting..."

    # Clean previous run artifacts
    rm -rf "$scene_dir/database.db" "$scene_dir/sparse" "$scene_dir/gsplat"

    local total_start=$(date +%s.%N)

    # ---- Stage 1: Global Matching ----
    local sfm_start=$(date +%s.%N)

    {
        echo "=== Stage 1: Global Matching ==="
        CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT_ROOT/global_corr/global_matching.py" \
            --image_dir "$scene_dir/images" \
            --output_dir "$scene_dir" 2>&1
    } > "$log_file" 2>&1

    # ---- Stage 2: FastMap ----
    {
        echo ""
        echo "=== Stage 2: FastMap ==="
        CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT_ROOT/fastmap/run.py" --headless \
            --database "$scene_dir/database.db" \
            --image_dir "$scene_dir/images" \
            --output_dir "$scene_dir" 2>&1
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

    local sfm_recall=$(python3 -c "import json; d=json.load(open('$sfm_eval_json')); m=list(d['per_scene'].values())[0]; print(m.get('recall_0.1m', 0))" 2>/dev/null || echo "0")
    local sfm_auc01=$(python3 -c "import json; d=json.load(open('$sfm_eval_json')); m=list(d['per_scene'].values())[0]; print(m.get('auc_0.1m', 0))" 2>/dev/null || echo "0")
    local sfm_auc05=$(python3 -c "import json; d=json.load(open('$sfm_eval_json')); m=list(d['per_scene'].values())[0]; print(m.get('auc_0.5m', 0))" 2>/dev/null || echo "0")

    echo "sfm_only,$scene_name,$prefix,$sfm_recall,$sfm_auc01,$sfm_auc05,$sfm_time,0,$sfm_time" >> "$CSV_FILE"

    # ---- Stage 3: Joint GS Optimization ----
    local joint_start=$(date +%s.%N)
    {
        echo ""
        echo "=== Stage 3: Joint GS Optimization ==="
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

    local joint_end=$(date +%s.%N)
    local joint_time=$(echo "$joint_end - $joint_start" | bc)

    local total_end=$(date +%s.%N)
    local total_time=$(echo "$total_end - $total_start" | bc)

    # ---- Evaluate joint-optimized poses ----
    # Optimized poses are stored as deltas in the gsplat checkpoint,
    # applied on top of the base SfM poses in sparse/0/.
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
            echo "WARNING: No checkpoint found in $ckpt_dir, evaluating base SfM poses."
            python "$SCRIPT_DIR/eval_eth3d_slam_poses.py" \
                --scene_dir "$scene_dir" \
                --sparse_dir "$scene_dir/sparse/0" \
                --output "$joint_eval_json" 2>&1
        fi
    } >> "$log_file" 2>&1

    local joint_recall=$(python3 -c "import json; d=json.load(open('$joint_eval_json')); m=list(d['per_scene'].values())[0]; print(m.get('recall_0.1m', 0))" 2>/dev/null || echo "0")
    local joint_auc01=$(python3 -c "import json; d=json.load(open('$joint_eval_json')); m=list(d['per_scene'].values())[0]; print(m.get('auc_0.1m', 0))" 2>/dev/null || echo "0")
    local joint_auc05=$(python3 -c "import json; d=json.load(open('$joint_eval_json')); m=list(d['per_scene'].values())[0]; print(m.get('auc_0.5m', 0))" 2>/dev/null || echo "0")

    echo "joint,$scene_name,$prefix,$joint_recall,$joint_auc01,$joint_auc05,$sfm_time,$joint_time,$total_time" >> "$CSV_FILE"

    # Copy logs and evaluation JSONs
    cp "$log_file" "$result_dir/" 2>/dev/null || true

    touch "$result_dir/done.marker"
    echo "[GPU $gpu] $scene_name: SfM(R@0.1=${sfm_recall}) Joint(R@0.1=${joint_recall}) time=${total_time}s"
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
    echo "STEP 2: Running SalientGS pipeline on all scenes"
    echo "================================================================"

    local all_scenes=()
    for scene_dir in "$WORK_DIR"/*/; do
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
# STEP 4: Generate summary
# =============================================================================
generate_summary() {
    echo ""
    echo "================================================================"
    echo "STEP 3: Generating Summary"
    echo "================================================================"

    # Run batch evaluation on the SfM results
    local sfm_summary="$RESULTS_DIR/sfm_summary.json"
    python "$SCRIPT_DIR/eval_eth3d_slam_poses.py" \
        --batch_dir "$WORK_DIR" \
        --sparse_subdir "sparse/0" \
        --output "$sfm_summary" 2>/dev/null || true

    echo ""
    echo "========================================"
    echo "SfM-Only Pose Evaluation"
    echo "========================================"
    python3 -c "
import json, sys
try:
    with open('$sfm_summary') as f:
        data = json.load(f)
    summary = data.get('summary', {})
    print(f\"{'Prefix':<15} {'Recall@0.1m':>12} {'AUC@0.1m':>10} {'AUC@0.5m':>10}\")
    print('-' * 50)
    prefixes = ['cables','camera','ceiling','desk','einstein','kidnap','large',
                'mannequin','motion','planar','plant','reflective','repetitive',
                'sfm','sofa','table','vicon']
    for p in prefixes:
        if p in summary:
            m = summary[p]
            print(f\"{p:<15} {m['recall_0.1m']:>12.1f} {m['auc_0.1m']:>10.1f} {m['auc_0.5m']:>10.1f}\")
    print('-' * 50)
    if 'average' in summary:
        m = summary['average']
        print(f\"{'Average':<15} {m['recall_0.1m']:>12.1f} {m['auc_0.1m']:>10.1f} {m['auc_0.5m']:>10.1f}\")
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
" 2>/dev/null || echo "  (summary generation failed)"

    echo ""
    echo "Results CSV: $CSV_FILE"
    echo "SfM summary: $sfm_summary"
    echo "Per-scene logs: $RESULTS_DIR/<scene>/run.log"
}

# =============================================================================
# MAIN
# =============================================================================
echo "========================================"
echo "ETH3D SLAM Pose Evaluation Runner"
echo "Date: $(date)"
echo "Zip dir: $ETH3D_ZIP_DIR"
echo "Work dir: $WORK_DIR"
echo "Results: $RESULTS_DIR"
echo "CSV: $CSV_FILE"
echo "GPUs: $NUM_GPUS"
echo "========================================"

extract_scenes
run_all_scenes
generate_summary

echo ""
echo "========================================"
echo "ETH3D SLAM evaluation complete!"
echo "========================================"
