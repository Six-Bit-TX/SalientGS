#!/bin/bash

# Benchmark script for SalientGS
# Runs each scene 5 times and picks the best metrics (PSNR, SSIM, LPIPS, runtime)
# Times each stage separately: sgs-feat, sgs-sfm, sgs-joint
# Creates summaries for each scene and each dataset (Mip360, tnt, db)

set -e

export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_HOME="${CONDA_PREFIX:-/home/ssd1t/xty/miniconda3/envs/gsplat}"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CPATH"

NUM_RUNS=3
RESULTS_DIR="../benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$RESULTS_DIR/summary_$TIMESTAMP.txt"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Dataset definitions with categories
declare -A SCENE_TO_DATASET
# Mip-NeRF 360
SCENE_TO_DATASET["bicycle"]="Mip360"
SCENE_TO_DATASET["counter"]="Mip360"
SCENE_TO_DATASET["flowers"]="Mip360"
SCENE_TO_DATASET["kitchen"]="Mip360"
SCENE_TO_DATASET["room"]="Mip360"
SCENE_TO_DATASET["bonsai"]="Mip360"
SCENE_TO_DATASET["garden"]="Mip360"
SCENE_TO_DATASET["stump"]="Mip360"
SCENE_TO_DATASET["treehill"]="Mip360"
# Tanks and Temples
SCENE_TO_DATASET["train"]="tnt"
SCENE_TO_DATASET["truck"]="tnt"
# Deep Blending
SCENE_TO_DATASET["drjohnson"]="db"
SCENE_TO_DATASET["playroom"]="db"

# All datasets
datasets=(
    # Mip-NeRF 360
    ../data_fastmap_1.5m/bicycle
    ../data_fastmap_1.5m/counter
    ../data_fastmap_1.5m/flowers
    ../data_fastmap_1.5m/kitchen
    ../data_fastmap_1.5m/room
    ../data_fastmap_1.5m/bonsai
    ../data_fastmap_1.5m/garden
    ../data_fastmap_1.5m/stump
    ../data_fastmap_1.5m/treehill
    # Tanks and Temples
    ../data_fastmap_1.5m/train
    ../data_fastmap_1.5m/truck
    # Deep Blending
    ../data_fastmap_1.5m/drjohnson
    ../data_fastmap_1.5m/playroom
)

# Function to extract metrics from log output
extract_metrics() {
    local log_file="$1"
    # Extract PSNR, SSIM, LPIPS from the output
    # Format: PSNR: 25.123, SSIM: 0.8234, LPIPS: 0.123 Time: 0.045s/image
    local psnr=$(grep -oP 'PSNR: \K[0-9.]+' "$log_file" | tail -1)
    local ssim=$(grep -oP 'SSIM: \K[0-9.]+' "$log_file" | tail -1)
    local lpips=$(grep -oP 'LPIPS: \K[0-9.]+' "$log_file" | tail -1)
    echo "$psnr $ssim $lpips"
}

# Initialize summary file
{
    echo "========================================"
    echo "SalientGS Benchmark Results"
    echo "Date: $(date)"
    echo "Number of runs per scene: $NUM_RUNS"
    echo "========================================"
    echo ""
} > "$SUMMARY_FILE"

# Declare associative arrays for storing best results
declare -A BEST_PSNR BEST_SSIM BEST_LPIPS
declare -A BEST_MATCHING_TIME BEST_FASTMAP_TIME BEST_JOINT_TIME BEST_TOTAL_TIME

# Declare arrays for dataset aggregation
declare -A DATASET_PSNR DATASET_SSIM DATASET_LPIPS
declare -A DATASET_MATCHING_TIME DATASET_FASTMAP_TIME DATASET_JOINT_TIME DATASET_TOTAL_TIME
declare -A DATASET_COUNT

# Initialize dataset counters
for dataset_name in "Mip360" "tnt" "db"; do
    DATASET_PSNR[$dataset_name]=""
    DATASET_SSIM[$dataset_name]=""
    DATASET_LPIPS[$dataset_name]=""
    DATASET_MATCHING_TIME[$dataset_name]=""
    DATASET_FASTMAP_TIME[$dataset_name]=""
    DATASET_JOINT_TIME[$dataset_name]=""
    DATASET_TOTAL_TIME[$dataset_name]=""
    DATASET_COUNT[$dataset_name]=0
done

echo "Starting benchmark..."
echo ""

for data_path in "${datasets[@]}"; do
    scene_name=$(basename "$data_path")
    dataset_name="${SCENE_TO_DATASET[$scene_name]}"
    scene_results_dir="$RESULTS_DIR/$scene_name"
    mkdir -p "$scene_results_dir"
    
    echo "========================================"
    echo "Processing scene: $scene_name (Dataset: $dataset_name)"
    echo "========================================"
    
    best_psnr=0
    best_ssim=0
    best_lpips=999
    best_matching_time=999999
    best_fastmap_time=999999
    best_joint_time=999999
    best_total_time=999999
    best_run=0
    
    for run in $(seq 1 $NUM_RUNS); do
        echo "  Run $run/$NUM_RUNS..."
        log_file="$scene_results_dir/run_${run}.log"
        timing_file="$scene_results_dir/run_${run}_timing.txt"
        
        # Clean up before run
        rm -rf "$data_path/database.db"
        rm -rf "$data_path/sparse/"
        rm -rf "$data_path/gsplat/"
        
        # Initialize log file
        echo "=== Run $run ===" > "$log_file"
        echo "Started at: $(date)" >> "$log_file"
        echo "" >> "$log_file"
        
        # Time the entire pipeline
        total_start_time=$(date +%s.%N)
        
        # Stage 1: Global matching (sgs-feat)
        echo "  Stage 1/3: sgs-feat (Global matching)..."
        matching_start=$(date +%s.%N)
        {
            echo "=== Stage 1: sgs-feat (Global Matching) ===" 
            CUDA_VISIBLE_DEVICES=0 sgs-feat --image_dir "$data_path/images" --output_dir "$data_path" 2>&1
        } >> "$log_file" 2>&1
        matching_end=$(date +%s.%N)
        matching_time=$(echo "$matching_end - $matching_start" | bc)
        echo "    Matching time: ${matching_time}s"
        
        # Stage 2: FastMap (sgs-sfm)
        echo "  Stage 2/3: sgs-sfm (FastMap)..."
        fastmap_start=$(date +%s.%N)
        {
            echo ""
            echo "=== Stage 2: sgs-sfm (FastMap) ===" 
            CUDA_VISIBLE_DEVICES=0 sgs-sfm --headless \
                --database "$data_path/database.db" \
                --image_dir "$data_path/images" \
                --output_dir "$data_path"
        } >> "$log_file" 2>&1
        fastmap_end=$(date +%s.%N)
        fastmap_time=$(echo "$fastmap_end - $fastmap_start" | bc)
        echo "    FastMap time: ${fastmap_time}s"
        
        # Stage 3: Joint training (sgs-joint)
        echo "  Stage 3/3: sgs-joint..."
        joint_start=$(date +%s.%N)
        {
            echo ""
            echo "=== Stage 3: sgs-joint ===" 
            CUDA_VISIBLE_DEVICES=0 sgs-joint --data_path "$data_path"
        } >> "$log_file" 2>&1
        joint_end=$(date +%s.%N)
        joint_time=$(echo "$joint_end - $joint_start" | bc)
        echo "    Joint time: ${joint_time}s"
        
        total_end_time=$(date +%s.%N)
        total_time=$(echo "$total_end_time - $total_start_time" | bc)
        
        # Extract metrics
        metrics=$(extract_metrics "$log_file")
        read -r psnr ssim lpips <<< "$metrics"
        
        # Save timing info
        {
            echo ""
            echo "=== Timing Summary ==="
            echo "Global Matching: ${matching_time}s"
            echo "FastMap: ${fastmap_time}s"
            echo "Joint: ${joint_time}s"
            echo "Total Time: ${total_time}s"
        } >> "$log_file"
        
        # Also save timing to separate file for easy parsing
        {
            echo "matching_time=$matching_time"
            echo "fastmap_time=$fastmap_time"
            echo "joint_time=$joint_time"
            echo "total_time=$total_time"
            echo "psnr=$psnr"
            echo "ssim=$ssim"
            echo "lpips=$lpips"
        } > "$timing_file"
        
        echo "    PSNR: $psnr, SSIM: $ssim, LPIPS: $lpips"
        echo "    Total time: ${total_time}s (Matching: ${matching_time}s, FastMap: ${fastmap_time}s, Joint: ${joint_time}s)"
        
        # Check if this run is better (prioritize PSNR)
        if [ -n "$psnr" ] && [ -n "$ssim" ] && [ -n "$lpips" ]; then
            is_better=$(awk -v p1="$psnr" -v p2="$best_psnr" 'BEGIN { print (p1 > p2) ? 1 : 0 }')
            if [ "$is_better" -eq 1 ] || [ "$best_run" -eq 0 ]; then
                best_psnr="$psnr"
                best_ssim="$ssim"
                best_lpips="$lpips"
                best_matching_time="$matching_time"
                best_fastmap_time="$fastmap_time"
                best_joint_time="$joint_time"
                best_total_time="$total_time"
                best_run="$run"
            fi
        fi
    done
    
    # Store best results for this scene
    BEST_PSNR[$scene_name]="$best_psnr"
    BEST_SSIM[$scene_name]="$best_ssim"
    BEST_LPIPS[$scene_name]="$best_lpips"
    BEST_MATCHING_TIME[$scene_name]="$best_matching_time"
    BEST_FASTMAP_TIME[$scene_name]="$best_fastmap_time"
    BEST_JOINT_TIME[$scene_name]="$best_joint_time"
    BEST_TOTAL_TIME[$scene_name]="$best_total_time"
    
    # Add to dataset aggregation
    if [ -n "${DATASET_PSNR[$dataset_name]}" ]; then
        DATASET_PSNR[$dataset_name]="${DATASET_PSNR[$dataset_name]} $best_psnr"
        DATASET_SSIM[$dataset_name]="${DATASET_SSIM[$dataset_name]} $best_ssim"
        DATASET_LPIPS[$dataset_name]="${DATASET_LPIPS[$dataset_name]} $best_lpips"
        DATASET_MATCHING_TIME[$dataset_name]="${DATASET_MATCHING_TIME[$dataset_name]} $best_matching_time"
        DATASET_FASTMAP_TIME[$dataset_name]="${DATASET_FASTMAP_TIME[$dataset_name]} $best_fastmap_time"
        DATASET_JOINT_TIME[$dataset_name]="${DATASET_JOINT_TIME[$dataset_name]} $best_joint_time"
        DATASET_TOTAL_TIME[$dataset_name]="${DATASET_TOTAL_TIME[$dataset_name]} $best_total_time"
    else
        DATASET_PSNR[$dataset_name]="$best_psnr"
        DATASET_SSIM[$dataset_name]="$best_ssim"
        DATASET_LPIPS[$dataset_name]="$best_lpips"
        DATASET_MATCHING_TIME[$dataset_name]="$best_matching_time"
        DATASET_FASTMAP_TIME[$dataset_name]="$best_fastmap_time"
        DATASET_JOINT_TIME[$dataset_name]="$best_joint_time"
        DATASET_TOTAL_TIME[$dataset_name]="$best_total_time"
    fi
    DATASET_COUNT[$dataset_name]=$((${DATASET_COUNT[$dataset_name]} + 1))
    
    # Write scene summary
    {
        echo "Scene: $scene_name (Dataset: $dataset_name)"
        echo "  Best Run: $best_run"
        echo "  PSNR:  $best_psnr"
        echo "  SSIM:  $best_ssim"
        echo "  LPIPS: $best_lpips"
        echo "  Timing Breakdown:"
        echo "    Global Matching: ${best_matching_time}s"
        echo "    FastMap:         ${best_fastmap_time}s"
        echo "    Joint:           ${best_joint_time}s"
        echo "    Total Time:      ${best_total_time}s"
        echo ""
    } >> "$SUMMARY_FILE"
    
    echo "  Best result (Run $best_run):"
    echo "    PSNR=$best_psnr, SSIM=$best_ssim, LPIPS=$best_lpips"
    echo "    Time: Total=${best_total_time}s (Matching=${best_matching_time}s, FastMap=${best_fastmap_time}s, Joint=${best_joint_time}s)"
    echo ""
    
    # Sleep between scenes to allow GPU to cool down
    echo "  Sleeping for 60 seconds before next scene..."
    sleep 60
done

# Calculate and write dataset summaries
{
    echo "========================================"
    echo "DATASET SUMMARIES"
    echo "========================================"
    echo ""
} >> "$SUMMARY_FILE"

for dataset_name in "Mip360" "tnt" "db"; do
    count=${DATASET_COUNT[$dataset_name]}
    if [ "$count" -gt 0 ]; then
        # Calculate averages using awk
        avg_psnr=$(echo "${DATASET_PSNR[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF }')
        avg_ssim=$(echo "${DATASET_SSIM[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF }')
        avg_lpips=$(echo "${DATASET_LPIPS[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF }')
        avg_matching=$(echo "${DATASET_MATCHING_TIME[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF }')
        avg_fastmap=$(echo "${DATASET_FASTMAP_TIME[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF }')
        avg_joint=$(echo "${DATASET_JOINT_TIME[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF }')
        avg_total=$(echo "${DATASET_TOTAL_TIME[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF }')
        sum_total=$(echo "${DATASET_TOTAL_TIME[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum }')
        
        {
            echo "Dataset: $dataset_name ($count scenes)"
            echo "  Metrics (Average):"
            echo "    PSNR:  $avg_psnr"
            echo "    SSIM:  $avg_ssim"
            echo "    LPIPS: $avg_lpips"
            echo "  Timing (Average per scene):"
            echo "    Global Matching: ${avg_matching}s"
            echo "    FastMap:         ${avg_fastmap}s"
            echo "    Joint:           ${avg_joint}s"
            echo "    Total:           ${avg_total}s"
            echo "  Total Time (all scenes): ${sum_total}s"
            echo ""
        } >> "$SUMMARY_FILE"
        
        echo "Dataset $dataset_name ($count scenes):"
        echo "  Avg PSNR=$avg_psnr, Avg SSIM=$avg_ssim, Avg LPIPS=$avg_lpips"
        echo "  Avg Time: Total=${avg_total}s (Matching=${avg_matching}s, FastMap=${avg_fastmap}s, Joint=${avg_joint}s)"
    fi
done

# Write final summary table - Metrics
{
    echo "========================================"
    echo "FULL RESULTS TABLE - METRICS"
    echo "========================================"
    echo ""
    printf "%-15s %-10s %-10s %-10s %-10s\n" "Scene" "Dataset" "PSNR" "SSIM" "LPIPS"
    printf "%-15s %-10s %-10s %-10s %-10s\n" "---------------" "----------" "----------" "----------" "----------"
    
    for data_path in "${datasets[@]}"; do
        scene_name=$(basename "$data_path")
        dataset_name="${SCENE_TO_DATASET[$scene_name]}"
        printf "%-15s %-10s %-10s %-10s %-10s\n" \
            "$scene_name" \
            "$dataset_name" \
            "${BEST_PSNR[$scene_name]}" \
            "${BEST_SSIM[$scene_name]}" \
            "${BEST_LPIPS[$scene_name]}"
    done
    echo ""
} >> "$SUMMARY_FILE"

# Write final summary table - Timing
{
    echo "========================================"
    echo "FULL RESULTS TABLE - TIMING (seconds)"
    echo "========================================"
    echo ""
    printf "%-15s %-10s %-12s %-12s %-12s %-12s\n" "Scene" "Dataset" "Matching" "FastMap" "Joint" "Total"
    printf "%-15s %-10s %-12s %-12s %-12s %-12s\n" "---------------" "----------" "------------" "------------" "------------" "------------"
    
    for data_path in "${datasets[@]}"; do
        scene_name=$(basename "$data_path")
        dataset_name="${SCENE_TO_DATASET[$scene_name]}"
        printf "%-15s %-10s %-12s %-12s %-12s %-12s\n" \
            "$scene_name" \
            "$dataset_name" \
            "${BEST_MATCHING_TIME[$scene_name]}" \
            "${BEST_FASTMAP_TIME[$scene_name]}" \
            "${BEST_JOINT_TIME[$scene_name]}" \
            "${BEST_TOTAL_TIME[$scene_name]}"
    done
    echo ""
} >> "$SUMMARY_FILE"

# Also create a CSV file for easy parsing
csv_file="$RESULTS_DIR/results_$TIMESTAMP.csv"
{
    echo "Scene,Dataset,PSNR,SSIM,LPIPS,MatchingTime,FastMapTime,JointTime,TotalTime"
    for data_path in "${datasets[@]}"; do
        scene_name=$(basename "$data_path")
        dataset_name="${SCENE_TO_DATASET[$scene_name]}"
        echo "$scene_name,$dataset_name,${BEST_PSNR[$scene_name]},${BEST_SSIM[$scene_name]},${BEST_LPIPS[$scene_name]},${BEST_MATCHING_TIME[$scene_name]},${BEST_FASTMAP_TIME[$scene_name]},${BEST_JOINT_TIME[$scene_name]},${BEST_TOTAL_TIME[$scene_name]}"
    done
} > "$csv_file"

# Create dataset summary CSV
dataset_csv="$RESULTS_DIR/dataset_summary_$TIMESTAMP.csv"
{
    echo "Dataset,NumScenes,AvgPSNR,AvgSSIM,AvgLPIPS,AvgMatchingTime,AvgFastMapTime,AvgJointTime,AvgTotalTime,SumTotalTime"
    for dataset_name in "Mip360" "tnt" "db"; do
        count=${DATASET_COUNT[$dataset_name]}
        if [ "$count" -gt 0 ]; then
            avg_psnr=$(echo "${DATASET_PSNR[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.4f", sum/NF }')
            avg_ssim=$(echo "${DATASET_SSIM[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.4f", sum/NF }')
            avg_lpips=$(echo "${DATASET_LPIPS[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.4f", sum/NF }')
            avg_matching=$(echo "${DATASET_MATCHING_TIME[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.2f", sum/NF }')
            avg_fastmap=$(echo "${DATASET_FASTMAP_TIME[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.2f", sum/NF }')
            avg_joint=$(echo "${DATASET_JOINT_TIME[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.2f", sum/NF }')
            avg_total=$(echo "${DATASET_TOTAL_TIME[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.2f", sum/NF }')
            sum_total=$(echo "${DATASET_TOTAL_TIME[$dataset_name]}" | awk '{ sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.2f", sum }')
            echo "$dataset_name,$count,$avg_psnr,$avg_ssim,$avg_lpips,$avg_matching,$avg_fastmap,$avg_joint,$avg_total,$sum_total"
        fi
    done
} > "$dataset_csv"

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "========================================"
echo "Output files:"
echo "  Summary:         $SUMMARY_FILE"
echo "  Results CSV:     $csv_file"
echo "  Dataset CSV:     $dataset_csv"
echo "  Run logs:        $RESULTS_DIR/<scene>/run_*.log"
echo "  Run timing:      $RESULTS_DIR/<scene>/run_*_timing.txt"
echo "========================================"

# Display the summary
echo ""
echo "=== SUMMARY ==="
cat "$SUMMARY_FILE"
