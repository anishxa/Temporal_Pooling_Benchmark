# Temporal Pooling Benchmark: Expanded Comparison Report

This report compares 6 pooling architectures across 6 speech Self-Supervised Learning (SSL) backbones on **E-DAIC** and **MODMA** datasets using **semi-fine-tuning** (learnable weighted sum of hidden layers).

## Dataset: EDAIC

### Backbone SSL Model: `data2vec-large`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Statistical Pooling | 0.7391 | 0.5000 | 0.6471 | 0.5000 | 0.8235 |
| Mean Pooling | 0.6087 | 0.4706 | 0.5490 | 0.6667 | 0.5882 |
| Bi-GRU + Attention | 0.6957 | 0.4615 | 0.7451 | 0.5000 | 0.7647 |
| Transformer Encoder | 0.5652 | 0.4444 | 0.6078 | 0.6667 | 0.5294 |
| Self-Attention | 0.2609 | 0.4138 | 0.5294 | 1.0000 | 0.0000 |
| NetVLAD | 0.2609 | 0.4138 | 0.5882 | 1.0000 | 0.0000 |

### Backbone SSL Model: `hubert-large`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Mean Pooling | 0.7826 | 0.6154 | 0.5882 | 0.6667 | 0.8235 |
| Self-Attention | 0.6522 | 0.5000 | 0.6373 | 0.6667 | 0.6471 |
| Statistical Pooling | 0.3913 | 0.4615 | 0.5490 | 1.0000 | 0.1765 |
| Bi-GRU + Attention | 0.3043 | 0.4286 | 0.5490 | 1.0000 | 0.0588 |
| NetVLAD | 0.2609 | 0.4138 | 0.6275 | 1.0000 | 0.0000 |
| Transformer Encoder | 0.7391 | 0.0000 | 0.5392 | 0.0000 | 1.0000 |

### Backbone SSL Model: `w2v2-robust`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.3043 | 0.4286 | 0.5882 | 1.0000 | 0.0588 |
| Mean Pooling | 0.2609 | 0.4138 | 0.5980 | 1.0000 | 0.0000 |
| Statistical Pooling | 0.2609 | 0.4138 | 0.5882 | 1.0000 | 0.0000 |
| Self-Attention | 0.2609 | 0.4138 | 0.5000 | 1.0000 | 0.0000 |
| Transformer Encoder | 0.2609 | 0.4138 | 0.5196 | 1.0000 | 0.0000 |
| NetVLAD | 0.2609 | 0.4138 | 0.4657 | 1.0000 | 0.0000 |

### Backbone SSL Model: `wavlm-base-plus`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Self-Attention | 0.8696 | 0.6667 | 0.7157 | 0.5000 | 1.0000 |
| NetVLAD | 0.8696 | 0.6667 | 0.6765 | 0.5000 | 1.0000 |
| Mean Pooling | 0.7826 | 0.6154 | 0.6569 | 0.6667 | 0.8235 |
| Bi-GRU + Attention | 0.7826 | 0.6154 | 0.6471 | 0.6667 | 0.8235 |
| Statistical Pooling | 0.7391 | 0.5714 | 0.7157 | 0.6667 | 0.7647 |
| Transformer Encoder | 0.6087 | 0.4706 | 0.6471 | 0.6667 | 0.5882 |

### Backbone SSL Model: `wavlm-large`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.7826 | 0.6154 | 0.7353 | 0.6667 | 0.8235 |
| Statistical Pooling | 0.7391 | 0.5714 | 0.6863 | 0.6667 | 0.7647 |
| Mean Pooling | 0.6522 | 0.5556 | 0.6569 | 0.8333 | 0.5882 |
| Self-Attention | 0.7826 | 0.5455 | 0.6078 | 0.5000 | 0.8824 |
| NetVLAD | 0.5652 | 0.5000 | 0.6569 | 0.8333 | 0.4706 |
| Transformer Encoder | 0.7391 | 0.0000 | 0.6961 | 0.0000 | 1.0000 |

### Backbone SSL Model: `xls-r-1b`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Statistical Pooling | 0.5652 | 0.5455 | 0.7451 | 1.0000 | 0.4118 |
| Mean Pooling | 0.6957 | 0.5333 | 0.6275 | 0.6667 | 0.7059 |
| Self-Attention | 0.4783 | 0.5000 | 0.5980 | 1.0000 | 0.2941 |
| Bi-GRU + Attention | 0.7391 | 0.5000 | 0.7059 | 0.5000 | 0.8235 |
| Transformer Encoder | 0.2609 | 0.4138 | 0.4510 | 1.0000 | 0.0000 |
| NetVLAD | 0.2609 | 0.4138 | 0.6275 | 1.0000 | 0.0000 |

## Dataset: MODMA

### Backbone SSL Model: `data2vec-large`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.9000 | 0.9091 | 0.8400 | 1.0000 | 0.8000 |
| Mean Pooling | 0.5000 | 0.6667 | 0.3200 | 1.0000 | 0.0000 |
| Statistical Pooling | 0.5000 | 0.6667 | 0.4000 | 1.0000 | 0.0000 |
| Self-Attention | 0.6000 | 0.6667 | 0.5200 | 0.8000 | 0.4000 |
| Transformer Encoder | 0.5000 | 0.6667 | 0.3200 | 1.0000 | 0.0000 |
| NetVLAD | 0.6000 | 0.6000 | 0.6400 | 0.6000 | 0.6000 |

### Backbone SSL Model: `hubert-large`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Mean Pooling | 0.9000 | 0.8889 | 0.8000 | 0.8000 | 1.0000 |
| Bi-GRU + Attention | 0.9000 | 0.8889 | 0.8000 | 0.8000 | 1.0000 |
| NetVLAD | 0.9000 | 0.8889 | 0.8000 | 0.8000 | 1.0000 |
| Statistical Pooling | 0.8000 | 0.7500 | 0.7200 | 0.6000 | 1.0000 |
| Self-Attention | 0.8000 | 0.7500 | 0.7200 | 0.6000 | 1.0000 |
| Transformer Encoder | 0.5000 | 0.6667 | 0.3600 | 1.0000 | 0.0000 |

### Backbone SSL Model: `w2v2-robust`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.6000 | 0.7143 | 0.5200 | 1.0000 | 0.2000 |
| Mean Pooling | 0.5000 | 0.6667 | 0.6400 | 1.0000 | 0.0000 |
| Statistical Pooling | 0.5000 | 0.6667 | 0.5000 | 1.0000 | 0.0000 |
| Self-Attention | 0.5000 | 0.6667 | 0.6400 | 1.0000 | 0.0000 |
| Transformer Encoder | 0.5000 | 0.6667 | 0.5600 | 1.0000 | 0.0000 |
| NetVLAD | 0.5000 | 0.0000 | 0.4600 | 0.0000 | 1.0000 |

### Backbone SSL Model: `wavlm-base-plus`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.9000 | 0.8889 | 0.8000 | 0.8000 | 1.0000 |
| Mean Pooling | 0.8000 | 0.7500 | 0.6800 | 0.6000 | 1.0000 |
| Statistical Pooling | 0.8000 | 0.7500 | 0.7200 | 0.6000 | 1.0000 |
| Self-Attention | 0.8000 | 0.7500 | 0.8000 | 0.6000 | 1.0000 |
| NetVLAD | 0.8000 | 0.7500 | 0.7600 | 0.6000 | 1.0000 |
| Transformer Encoder | 0.5000 | 0.6667 | 0.3600 | 1.0000 | 0.0000 |

### Backbone SSL Model: `wavlm-large`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| NetVLAD | 0.9000 | 0.8889 | 0.8000 | 0.8000 | 1.0000 |
| Self-Attention | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 |
| Mean Pooling | 0.8000 | 0.7500 | 0.7600 | 0.6000 | 1.0000 |
| Statistical Pooling | 0.5000 | 0.6667 | 0.2800 | 1.0000 | 0.0000 |
| Transformer Encoder | 0.6000 | 0.6667 | 0.6400 | 0.8000 | 0.4000 |
| Bi-GRU + Attention | 0.7000 | 0.6667 | 0.5600 | 0.6000 | 0.8000 |

### Backbone SSL Model: `xls-r-1b`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.8000 | 0.8000 | 0.7600 | 0.8000 | 0.8000 |
| Mean Pooling | 0.8000 | 0.7500 | 0.6800 | 0.6000 | 1.0000 |
| Statistical Pooling | 0.8000 | 0.7500 | 0.6000 | 0.6000 | 1.0000 |
| Self-Attention | 0.8000 | 0.7500 | 0.6000 | 0.6000 | 1.0000 |
| Transformer Encoder | 0.5000 | 0.6667 | 0.3600 | 1.0000 | 0.0000 |
| NetVLAD | 0.7000 | 0.6667 | 0.6000 | 0.6000 | 0.8000 |

