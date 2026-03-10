#!/bin/bash
# Prepare k-retrieval ablation data for different top_k_pairs values
# Creates separate data directories and runs the full SfM pipeline
# (global_matching + fastmap) with different k values

set -e

export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_HOME="${CONDA_PREFIX:-/data16t/xty/miniconda3/envs/gsplat}"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CPATH"

DATA_SRC="/data16t/xty/FirstGS/data_fastmap_1.5m"
DATA_ROOT="/data16t/xty/FirstGS"

MIP360_SCENES=(bicycle bonsai counter flowers garden kitchen room stump treehill)
K_VALUES=(5 10 20)

run_sfm_for_k() {
    local scene="$1"
    local k="$2"
    local gpu="$3"
    local dst_dir="$DATA_ROOT/data_retrieval_k${k}/$scene"

    echo "[GPU $gpu] Starting SfM for $scene with k=$k..."

    mkdir -p "$dst_dir"

    # Symlink images directory
    if [ ! -L "$dst_dir/images" ] && [ ! -d "$dst_dir/images" ]; then
        ln -s "$DATA_SRC/$scene/images" "$dst_dir/images"
    fi

    # Skip if sparse data already exists
    if [ -f "$dst_dir/sparse/0/cameras.bin" ] || [ -f "$dst_dir/sparse/0/cameras.txt" ]; then
        echo "[GPU $gpu] $scene k=$k: SfM data already exists, skipping."
        return 0
    fi

    # Clean previous partial runs
    rm -rf "$dst_dir/database.db" "$dst_dir/sparse/"

    # Stage 1: Global matching with specific k
    echo "[GPU $gpu] $scene k=$k: Global matching..."
    CUDA_VISIBLE_DEVICES=$gpu python global_corr/global_matching.py \
        --image_dir "$dst_dir/images" \
        --output_dir "$dst_dir" \
        --top_k_pairs "$k" \
        --k_neighbors "$k"

    # Stage 2: FastMap
    echo "[GPU $gpu] $scene k=$k: FastMap..."
    CUDA_VISIBLE_DEVICES=$gpu python fastmap/run.py --headless \
        --database "$dst_dir/database.db" \
        --image_dir "$dst_dir/images" \
        --output_dir "$dst_dir"

    echo "[GPU $gpu] $scene k=$k: SfM complete."
}

echo "Preparing k-retrieval ablation data..."
echo "K values: ${K_VALUES[*]}"
echo "Scenes: ${MIP360_SCENES[*]}"
echo ""

NUM_GPUS=4

for k in "${K_VALUES[@]}"; do
    echo "========================================="
    echo "Processing k=$k"
    echo "========================================="

    job_idx=0
    for scene in "${MIP360_SCENES[@]}"; do
        gpu=$((job_idx % NUM_GPUS))
        run_sfm_for_k "$scene" "$k" "$gpu" &
        job_idx=$((job_idx + 1))

        if [ $((job_idx % NUM_GPUS)) -eq 0 ]; then
            echo "Waiting for batch to complete..."
            wait
            echo "Batch complete."
        fi
    done
    wait
    echo "All scenes for k=$k complete."
    echo ""
done

echo "All k-retrieval data preparation complete!"
