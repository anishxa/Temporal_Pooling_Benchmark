# DEPOOL: Expanded Comparison Report

This report summarizes the pooling architecture comparison across 6 speech SSL backbones on E-DAIC and MODMA.

## 1. Primary Comparison Grid (Single-Seed, matching Tables II & III in Paper)

No single-seed run results found.


## 2. Seed Sensitivity Analysis (Multi-Seed Averages, matching Table IV in Paper)

This table summarizes the mean and standard deviation across 3 random seeds (13, 42, 87) for the focal configurations, computed dynamically from authentic model prediction files:

### Dataset: EDAIC

#### Backbone SSL Model: `wavlm-base-plus`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.4783 ± 0.0939 | 0.4051 ± 0.0469 | 0.5980 ± 0.0577 | 0.6667 ± 0.0000 | 0.4118 ± 0.1271 |
| Transformer Encoder | 0.3913 ± 0.1845 | 0.4187 ± 0.0070 | 0.6046 ± 0.0092 | 0.8333 ± 0.2357 | 0.2353 ± 0.3328 |

#### Backbone SSL Model: `w2v2-robust`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.5652 ± 0.0000 | 0.3750 ± 0.0000 | 0.5245 ± 0.0139 | 0.5000 ± 0.0000 | 0.5882 ± 0.0000 |
| Transformer Encoder | 0.3188 ± 0.0820 | 0.4028 ± 0.0155 | 0.5703 ± 0.0663 | 0.8889 ± 0.1571 | 0.1176 ± 0.1664 |

### Dataset: MODMA

#### Backbone SSL Model: `wavlm-base-plus`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.5333 ± 0.0471 | 0.1111 ± 0.1571 | 0.5467 ± 0.1611 | 0.0667 ± 0.0943 | 1.0000 ± 0.0000 |
| Transformer Encoder | 0.5333 ± 0.0471 | 0.1111 ± 0.1571 | 0.5867 ± 0.0377 | 0.0667 ± 0.0943 | 1.0000 ± 0.0000 |

#### Backbone SSL Model: `w2v2-robust`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.5000 ± 0.0000 | 0.6667 ± 0.0000 | 0.6133 ± 0.0822 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Transformer Encoder | 0.5000 ± 0.0000 | 0.6667 ± 0.0000 | 0.4267 ± 0.1181 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |

