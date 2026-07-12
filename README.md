# Chest X-Ray Pneumonia Classification

Classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using transfer
learning with CNNs, with explainability via Grad-CAM and a systematic
comparison of fine-tuning strategies and architectures.

## Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
— 5,856 pediatric chest X-ray images (anterior-posterior), labeled NORMAL or
PNEUMONIA, split into train/val/test sets.

**Note on the validation set**: the original Kaggle split's validation folder
contains only 16 images (8 per class). This causes validation accuracy to
swing dramatically between epochs (e.g. jumping to 100% then back down) —
not a sign of instability, just an artifact of too small a sample to be
statistically meaningful. All reported final results use the **test set**
(624 images), which gives a reliable signal.

## Approach

Three experiments were run to systematically compare training strategies:

1. **Full fine-tuning** (`model_ft`) — ResNet50 pretrained on ImageNet, all
   layers unfrozen and fine-tuned on chest X-rays.
2. **Frozen feature extractor** (`model_conv`) — ResNet50 with all
   convolutional layers frozen; only the final classification layer trained.
3. **Architecture comparison** — ResNet50, EfficientNet-B0, and DenseNet121,
   each trained with a frozen backbone (final layer only) for a fair,
   controlled comparison between architectures.

All models use `CrossEntropyLoss`, SGD optimizer, and a step learning-rate
scheduler. The architecture comparison additionally uses **class-weighted
loss** to address the dataset's class imbalance (more PNEUMONIA than NORMAL
images).

## Results

### Primary model: full fine-tuned ResNet50 (test set, 624 images)
| Metric | Value |
|---|---|
| Accuracy | 0.9215 |
| Precision | 0.9128 |
| Recall | **0.9667** |
| Specificity | 0.8462 |
| F1 Score | 0.9390 |
| ROC AUC | 0.9683 |

Confusion matrix: 377 true positives, 13 false negatives, 198 true negatives,
36 false positives.

### Fine-tuning strategy comparison (validation accuracy during training)
| Model | Best Val Acc | Notes |
|---|---|---|
| Full fine-tune (ResNet50) | 1.0000* | All layers trained |
| Frozen feature extractor (ResNet50) | 0.9375* | Only final layer trained |

*Val accuracy is on the 16-image validation set — see note above on
reliability; the gap in **training** accuracy (99.4% vs 93.5%) is the more
meaningful signal here, and confirms full fine-tuning learns considerably
better task-specific features.

### Architecture comparison (frozen backbone, test set)
| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Specificity |
|---|---|---|---|---|---|---|
| **DenseNet121** | 0.800 | 0.766 | 0.979 | **0.859** | **0.934** | 0.500 |
| EfficientNet-B0 | 0.776 | 0.747 | 0.969 | 0.844 | 0.911 | 0.453 |
| ResNet50 (frozen) | 0.728 | 0.701 | 0.982 | 0.818 | 0.903 | 0.303 |

DenseNet121 was the strongest architecture under frozen-backbone conditions.
**Important**: none of these three match the fully fine-tuned ResNet50 above
(F1 0.86 vs 0.94) — they use a different, more constrained training strategy
(frozen backbone), so this comparison answers "which architecture works best
with limited compute/frozen features," not "which model is best overall."

## Key findings

- **Recall over precision, by design**: for a pneumonia screening tool,
  missing a real case (false negative) is more costly than a false alarm.
  The primary model reflects this priority (Recall 0.967 > Precision 0.913),
  catching 377 of 390 true pneumonia cases.
- **Full fine-tuning meaningfully outperforms frozen-backbone transfer
  learning** on this task — domain-specific features (X-ray texture patterns)
  differ enough from ImageNet's natural-image features that unfreezing the
  backbone provides a real, non-trivial improvement.
- **Grad-CAM revealed a spurious correlation risk**: on several false-positive
  cases (true NORMAL, predicted PNEUMONIA), the model's attention concentrated
  on non-diagnostic regions — image corner markers and shoulder/collarbone
  areas — rather than lung tissue. This suggests the model may partly rely on
  incidental acquisition artifacts rather than purely clinical features, a
  known and important risk in medical imaging models trained on datasets with
  inconsistent acquisition conditions across sources.
- **Frozen-backbone specificity was consistently low** (0.30–0.50 across all
  three architectures) — with only the final layer trainable, models struggle
  to correctly reject NORMAL cases, likely because ImageNet features aren't
  discriminative enough for this fine-grained medical distinction without
  further adaptation.

## Explainability
Grad-CAM was used to visualize which regions of each X-ray the model
attended to when making its prediction, for both correct and incorrect
predictions — see the notebook for the full visualization grid.

## Files
- `notebooks/chest-xray-pneumonia-classification.ipynb`` — full pipeline:
data loading, training (both strategies), evaluation, Grad-CAM,
and architecture comparison
- `requirements.txt`
- `README.md`

## Setup
```bash
pip install -r requirements.txt
```

## Key learnings
- Always check validation set size before trusting per-epoch validation
  metrics — a 16-image val set produces noisy, unreliable signal even for a
  genuinely well-performing model.
- Explainability tools like Grad-CAM aren't just for presentation — they
  surfaced a real, actionable finding (potential reliance on non-clinical
  image artifacts) that pure metrics alone would have missed entirely.
- Fine-tuning strategy (frozen vs full) can matter more than architecture
  choice — a fully fine-tuned ResNet50 outperformed frozen-backbone
  DenseNet121, despite DenseNet's typically stronger baseline performance.

## Credits
Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
(Kermany et al., Mendeley Data). Pretrained models via `torchvision.models`.
