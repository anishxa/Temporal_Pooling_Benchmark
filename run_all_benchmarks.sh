#!/bin/bash
set -e

# List of models
models=("wavlm-base-plus" "wavlm-large" "xls-r-1b" "hubert-large" "w2v2-robust" "data2vec-large")

# List of datasets
datasets=("edaic" "modma")

echo "=========================================================================="
echo "          STARTING TEMPORAL POOLING BENCHMARK PIPELINE RUN"
echo "=========================================================================="

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------------"
    echo "Processing Dataset: ${dataset} | Model: ${model}"
    echo "--------------------------------------------------------------------------"
    
    # 1. Feature Extraction (All layers)
    echo "[Step 1/2] Extracting all-layer features..."
    python3 extract_features.py --model "$model" --dataset "$dataset" --batch_size 16
    
    # 2. Downstream Pooling Classifier Benchmark
    echo "[Step 2/2] Running downstream pooling benchmark..."
    python3 run_benchmark.py --model "$model" --dataset "$dataset" --epochs 40
    
  done
done

# 3. Consolidate results into a single report
echo ""
echo "=========================================================================="
echo "          CONSOLIDATING FINAL RESULTS AND METRICS"
echo "=========================================================================="
python3 consolidate_results.py

echo "Temporal Pooling Benchmark Expansion pipeline execution finished successfully!"
