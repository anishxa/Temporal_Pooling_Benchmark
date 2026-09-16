# DEPOOL: Expanded Comparison Report

This report summarizes the pooling architecture comparison across 6 speech SSL backbones on E-DAIC and MODMA.

## 1. Primary Comparison Grid (Single-Seed, matching Tables II & III in Paper)

No single-seed run results found.


## 2. Seed Sensitivity Analysis (Multi-Seed Averages, matching Table IV in Paper)

This table summarizes the mean and standard deviation across 3 random seeds (13, 42, 87) for the focal configurations:

### Dataset: EDAIC

#### Backbone SSL Model: `w2v2-robust`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Transformer Encoder | 0.3043 ± 0.0870 | 0.4056 ± 0.0164 | 0.5576 ± 0.0710 | 0.9167 ± 0.1667 | 0.0882 ± 0.1765 |
| Bi-GRU + Attention | 0.5000 ± 0.1304 | 0.3884 ± 0.0268 | 0.5404 ± 0.0347 | 0.6250 ± 0.2500 | 0.4559 ± 0.2647 |

#### Backbone SSL Model: `wavlm-base-plus`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.5543 ± 0.1788 | 0.4576 ± 0.1151 | 0.6103 ± 0.0627 | 0.6667 ± 0.0000 | 0.5147 ± 0.2419 |
| Transformer Encoder | 0.4457 ± 0.2141 | 0.4317 ± 0.0269 | 0.6152 ± 0.0232 | 0.7917 ± 0.2500 | 0.3235 ± 0.3767 |

### Dataset: MODMA

#### Backbone SSL Model: `w2v2-robust`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.5250 ± 0.0500 | 0.6786 ± 0.0238 | 0.5900 ± 0.0945 | 1.0000 ± 0.0000 | 0.0500 ± 0.1000 |

#### Backbone SSL Model: `wavlm-base-plus`

| Pooling Architecture | Accuracy | F1 Score | ROC AUC | Sensitivity (MDD) | Specificity (HC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bi-GRU + Attention | 0.6250 ± 0.1893 | 0.3056 ± 0.4194 | 0.6100 ± 0.2049 | 0.2500 ± 0.3786 | 1.0000 ± 0.0000 |
| Transformer Encoder | 0.5250 ± 0.0500 | 0.2500 ± 0.3191 | 0.5300 ± 0.1194 | 0.3000 ± 0.4761 | 0.7500 ± 0.5000 |

