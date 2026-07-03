# DEPOOL: Temporal Pooling Benchmark
This project compares different ways to group audio segments for detecting depression. We extract speech representations from pre-trained Self-Supervised Learning (SSL) models, and then we test different deep learning models to see which one works best for summarizing an entire 15-minute clinical interview.

This benchmark supports **6 SSL models** and **2 datasets** (E-DAIC and MODMA), and incorporates **semi-fine-tuning** via learnable layer aggregation.

---

## 1. Supported SSL Models

The benchmark evaluates 6 state-of-the-art speech SSL models, verifying that pre-training objective (rather than parameter scale) determines zero-shot transfer performance:
1. **WavLM Base-Plus** (`microsoft/wavlm-base-plus`) - 13 layers, 768-d (**94M parameters**)
2. **WavLM Large** (`microsoft/wavlm-large`) - 25 layers, 1024-d (**315M parameters**)
3. **XLS-R-1B** (`facebook/wav2vec2-xls-r-1b`) - 49 layers, 1280-d (**1B parameters**)
4. **HuBERT Large** (`facebook/hubert-large-ls960-ft`) - 25 layers, 1024-d (**315M parameters**)
5. **Wav2Vec2 Robust** (`facebook/wav2vec2-large-robust`) - 25 layers, 1024-d (**315M parameters**)
6. **Data2Vec Audio Large** (`facebook/data2vec-audio-large-960h`) - 25 layers, 1024-d (**315M parameters**)

---

## 2. Downstream Pooling Architectures

We test 6 different sequence models:
* **Mean Pooling**
* **Statistical Pooling** (Mean + Std + Min + Max)
* **Self-Attention**
* **Bi-GRU with Attention**
* **NetVLAD**
* **Transformer Encoder**

---

## 3. Semi-Fine-Tuning Methodology

Instead of utilizing a single hardcoded layer, we implement **semi-fine-tuning** using a learnable weighted sum of hidden states (matching the featurizer behavior in benchmarks like SUPERB / EMO-SUPERB). 

For each segment, we extract all hidden layers, pool them across the time dimension using `mean(dim=1)`, and save the representation as a `[num_layers, hidden_dim]` array. During downstream training, a learnable **`Featurizer`** module aggregates the layers dynamically:

$$Representation = \sum_{i=1}^{L} \text{Softmax}(w_i) \cdot Layer_i$$

where $w_i$ are learnable layer weights. The aggregated representation is projected to a uniform `projector_dim = 256` before being fed into the pooling architecture.

---

## 4. How to Run the Pipeline

### Unified Runs

1. **Run the full benchmark**:
   To automatically run feature extraction and benchmark training for all 6 models, 2 datasets, 6 heads, and 3 pooling variants:
   ```bash
   ./run_all_benchmarks.sh
   ```

2. **Run a fast subset (Option B)**:
   To run a streamlined subset of the grid (evaluating WavLM Base-Plus and Wav2Vec2 Robust on both datasets using Bi-GRU and Transformer Encoder heads across seeds 13, 42, and 87):
   ```bash
   ./run_fast_subset.sh
   ```

### Running Individual Configurations

If you want to run a specific model or dataset combination individually:

1. **Extract all-layer features**:
   ```bash
   python3 codes/extract_features.py --model wavlm-base-plus --dataset edaic
   ```
   * Choices for `--model`: `wavlm-base-plus`, `wavlm-large`, `xls-r-1b`, `hubert-large`, `w2v2-robust`, `data2vec-large`.
   * Choices for `--dataset`: `edaic`, `modma`.

2. **Train and evaluate**:
   ```bash
   python3 codes/run_benchmark.py --model wavlm-base-plus --dataset edaic --epochs 40
   ```

3. **Consolidate results**:
   To compile all individual run metrics into the final markdown report (`output/temporal_pooling_summary.md`):
   ```bash
   python3 codes/consolidate_results.py
   ```

---

## 5. Directory Structure

The project repository is structured as follows:
*   **`codes/`**: Contains Python scripts for models (`models.py`), dataset loader (`dataset.py`), feature extraction (`extract_features.py`), running classifier benchmarks (`run_benchmark.py`), and result consolidation (`consolidate_results.py`).
*   **`data/`**: Holds metadata tables for segment splits (`utterance_table_edaic_segmented_split.csv` and `utterance_table_modma_segmented_split.csv`).
*   **`output/`**: Contains individual metric CSV files and the compiled summary report (`temporal_pooling_summary.md`).
*   **`features/`**: (Auto-generated) Stores extracted intermediate speech representations.

---

## 6. Output Results

* Individual metrics are saved in `output/pooling_benchmark_{dataset}_{model}.csv`.
* The final consolidated report is compiled at `output/temporal_pooling_summary.md` and `output/temporal_pooling_all_results.csv`.
