#!/bin/bash
# Option B (Focal Pair, 2-Head, 3-Seed, 3-Variant Grid) Orchestration Script
set -e

cd "$(dirname "$0")"
LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"

echo "=========================================================================="
echo "   FAST SUBSET ORCHESTRATION (OPTION B)"
echo "   Models: wavlm-base-plus, w2v2-robust"
echo "   Datasets: edaic, modma"
echo "   Heads: Bi-GRU + Attention, Transformer Encoder"
echo "   Variants: learned, uniform, fixed_sweep"
echo "   Seeds: 13 42 87"
echo "=========================================================================="

models=("wavlm-base-plus" "w2v2-robust")
datasets=("edaic" "modma")
heads=("Bi-GRU + Attention" "Transformer Encoder")
variants=("learned" "uniform" "fixed_sweep")
seeds=(13 42 87)

# Ensure features are cached (this step runs if not already cached)
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    if [ ! -f "features/${dataset}/${model}/X_test_all_layers.npy" ]; then
        echo "[GPU] Extracting features for ${dataset} / ${model} (fp16)..."
        batch=16
        if [ "$model" == "wavlm-base-plus" ]; then batch=32; fi
        python3 codes/extract_features.py --model "$model" --dataset "$dataset" --batch_size $batch --fp16 > "$LOG_DIR/extract_${dataset}_${model}.log" 2>&1
    else
        echo "[INFO] Features already cached for ${dataset} / ${model}"
    fi
  done
done

echo ""
echo "Generating list of benchmark jobs (8 cells * 3 variants * 3 seeds = 72 jobs)..."
JOBS_FILE="$LOG_DIR/jobs_list.txt"
rm -f "$JOBS_FILE"

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for head in "${heads[@]}"; do
      for variant in "${variants[@]}"; do
        for seed in "${seeds[@]}"; do
          head_safe=$(echo "$head" | sed -e 's/ /_/g' -e 's/+/plus/g')
          log_file="$LOG_DIR/bench_${dataset}_${model}_${head_safe}_${variant}_seed${seed}.log"
          
          echo "python3 codes/run_benchmark.py --model \"$model\" --dataset \"$dataset\" --head \"$head_safe\" --seed $seed --featurizer_type \"$variant\" --epochs 40 > \"$log_file\" 2>&1" >> "$JOBS_FILE"
        done
      done
    done
  done
done

echo "Starting jobs in parallel (max 4 concurrent processes)..."
xargs -P 4 -I {} sh -c "{}" < "$JOBS_FILE"
echo "All training runs complete!"

echo ""
echo "Consolidating results..."
python3 codes/consolidate_results.py

echo ""
echo "Running full statistical post-processing (Tasks 3, 4, 5, 6, 7, 8)..."
python3 codes/statistical_analysis.py

echo "Done! Check reports and CSV files in output/"
