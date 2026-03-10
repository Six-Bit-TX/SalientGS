#!/bin/bash
# Runtime Comparison Script for Global Matching + FastMap
# Times global_matching and fastmap stages on different dataset sizes
# Uses Courthouse dataset (1000+ images) with sequential sampling
#
# Usage: 
#   bash run_fast_runtime.sh                    # Run on all sizes (250, 500, 750, 1000)
#   bash run_fast_runtime.sh 500                # Run on specific size only
#
# Output: runtime_results/Courthouse_n<size>/fast_runtime_results.json

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../runtime_results"
mkdir -p "$OUTPUT_DIR"

# Dataset sizes to test (sequential sampling: first N images)
SIZES=(250 500 750 1000)

# Courthouse dataset path (1000+ images available)
DATA_PATH="/home/xiongti1/unify/glogs/TNT_GOF/Courthouse"

# Parse arguments
SIZE_FILTER="${1:-all}"

echo "=============================================="
echo "Runtime: Global Matching + FastMap"
echo "=============================================="
echo "Dataset: Courthouse (sequential sampling)"
echo "Output directory: $OUTPUT_DIR"
echo "Size filter: $SIZE_FILTER"
echo ""

# Check if dataset exists
if [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] Courthouse dataset not found: $DATA_PATH"
    exit 1
fi

# Count available images
ACTUAL_COUNT=$(find "$DATA_PATH/images" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.JPG" -o -name "*.PNG" \) 2>/dev/null | wc -l)
echo "Available images: $ACTUAL_COUNT"
echo ""

# Run comparison for each size
for size in "${SIZES[@]}"; do
    # Apply size filter
    if [ "$SIZE_FILTER" != "all" ] && [ "$size" != "$SIZE_FILTER" ]; then
        continue
    fi

    echo ""
    echo "----------------------------------------------"
    echo "Running: Courthouse | $size images (sequential)"
    echo "----------------------------------------------"

    # Check if dataset has enough images
    if [ "$ACTUAL_COUNT" -lt "$size" ]; then
        echo "[WARN] Dataset has only $ACTUAL_COUNT images, less than requested $size. Skipping."
        continue
    fi

    # Prepare output directory for this size
    SIZE_OUTPUT_DIR="$OUTPUT_DIR/Courthouse_n${size}"
    mkdir -p "$SIZE_OUTPUT_DIR"

    # Create a separate image folder with first N images (sequential sampling)
    SIZE_IMAGE_DIR="$SIZE_OUTPUT_DIR/images"
    rm -rf "$SIZE_IMAGE_DIR"
    mkdir -p "$SIZE_IMAGE_DIR"

    echo "[PREP] Copying first $size images to $SIZE_IMAGE_DIR..."
    find "$DATA_PATH/images" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.JPG" -o -name "*.PNG" \) | sort | head -n "$size" | while read img; do
        cp "$img" "$SIZE_IMAGE_DIR/"
    done
    COPIED_COUNT=$(find "$SIZE_IMAGE_DIR" -type f | wc -l)
    echo "[PREP] Copied $COPIED_COUNT images"

    # Clean up previous database
    rm -rf "$SIZE_OUTPUT_DIR/database.db"

    # --- Time global_matching ---
    echo "[START] global_matching ($size images)..."
    GM_START=$(date +%s%N)

    python "$SCRIPT_DIR/../global_corr/global_matching.py" \
        --image_dir "$SIZE_IMAGE_DIR" \
        --output_dir "$SIZE_OUTPUT_DIR"

    GM_END=$(date +%s%N)
    GM_ELAPSED=$(( (GM_END - GM_START) / 1000000 ))  # milliseconds
    GM_ELAPSED_SEC=$(echo "scale=2; $GM_ELAPSED / 1000" | bc)
    echo "[DONE] global_matching: ${GM_ELAPSED_SEC}s"

    # --- Time fastmap ---
    echo "[START] fastmap ($size images)..."
    FM_START=$(date +%s%N)

    python "$SCRIPT_DIR/../fastmap/run.py" --headless \
        --database "$SIZE_OUTPUT_DIR/database.db" \
        --image_dir "$SIZE_IMAGE_DIR" \
        --output_dir "$SIZE_OUTPUT_DIR"

    FM_END=$(date +%s%N)
    FM_ELAPSED=$(( (FM_END - FM_START) / 1000000 ))  # milliseconds
    FM_ELAPSED_SEC=$(echo "scale=2; $FM_ELAPSED / 1000" | bc)
    echo "[DONE] fastmap: ${FM_ELAPSED_SEC}s"

    # Total time
    TOTAL_ELAPSED=$(( GM_ELAPSED + FM_ELAPSED ))
    TOTAL_ELAPSED_SEC=$(echo "scale=2; $TOTAL_ELAPSED / 1000" | bc)
    echo "[TOTAL] ${TOTAL_ELAPSED_SEC}s (global_matching: ${GM_ELAPSED_SEC}s + fastmap: ${FM_ELAPSED_SEC}s)"

    # Save results to JSON
    cat > "$SIZE_OUTPUT_DIR/fast_runtime_results.json" <<EOF
{
    "dataset": "Courthouse",
    "num_images": $size,
    "sampling": "sequential",
    "global_matching_ms": $GM_ELAPSED,
    "global_matching_sec": $GM_ELAPSED_SEC,
    "fastmap_ms": $FM_ELAPSED,
    "fastmap_sec": $FM_ELAPSED_SEC,
    "total_ms": $TOTAL_ELAPSED,
    "total_sec": $TOTAL_ELAPSED_SEC
}
EOF

    echo "[SAVED] $SIZE_OUTPUT_DIR/fast_runtime_results.json"
    echo "[DONE] Courthouse @ $size images"

    # Clean up gsplat folder
    rm -rf "$SIZE_OUTPUT_DIR/gsplat/"

    # Cool down before next size
    echo "[WAIT] Sleeping 60 seconds before next size..."
    sleep 60
done

echo ""
echo "=============================================="
echo "All fast runtime measurements completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
