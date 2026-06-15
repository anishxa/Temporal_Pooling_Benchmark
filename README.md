# Temporal Pooling Benchmark

This project compares different ways to group audio segments for detecting depression. We use WavLM to get audio features for 10 second clips, and then we test different deep learning models to see which one works best for summarizing an entire 15 minute clinical interview.

## How It Works

1. **Preprocessing**: We cut the long clinical audio files into small 10 second chunks.
2. **Feature Extraction**: We pass each chunk through WavLM to get a 768 dimensional feature vector.
3. **Temporal Pooling**: We group all the chunks for a single patient back together chronologically. Then, we pass them through a sequence model (like a Transformer or NetVLAD) to get a final prediction.

## Models Evaluated

We test 6 different models:
* Mean Pooling
* Statistical Pooling
* Self Attention
* Bi-GRU with Attention
* NetVLAD
* Transformer Encoder

## How to Run

First, extract the WavLM features locally:
```bash
python3 extract_features.py
```

Then, train all 6 models and generate the final results table:
```bash
python3 run_benchmark.py
```

## Results

Simple models like Statistical Pooling actually work best for small datasets, while massive models like Transformers tend to overfit.

| Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Mean Pooling | 0.8696 | 0.6667 | 0.6471 | 0.5000 | 1.0000 |
| Statistical Pooling | 0.8696 | 0.6667 | 0.7059 | 0.5000 | 1.0000 |
| Bi-GRU + Attention | 0.7826 | 0.6154 | 0.6471 | 0.6667 | 0.8235 |
| NetVLAD | 0.7391 | 0.5714 | 0.6176 | 0.6667 | 0.7647 |
| Self Attention | 0.6957 | 0.5333 | 0.6275 | 0.6667 | 0.7059 |
| Transformer Encoder | 0.6087 | 0.5263 | 0.6863 | 0.8333 | 0.5294 |
