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
