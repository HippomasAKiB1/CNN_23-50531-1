# CNN for Multi-Label Remote Sensing Scene Classification

**Name**: Akib Hasan Pyil

**ID**: 23-50531-1  

**Course**: Computer Vision and Pattern Recognition (section c)

**Faculty**: Rifath Mahmud

**Institution**: American International University-Bangladesh (AIUB)

**Dataset**: MLRSNet — Multi-Label High Spatial Resolution Remote Sensing Dataset  


---

## What This Is

A custom CNN built from scratch for multi-label classification on satellite images. The dataset is MLRSNet — 109,161 images across 46 scene categories, each tagged with multiple semantic labels out of 60 possible ones (things like `buildings`, `road`, `water`, `trees`, `cars`).

The main question I was testing: does batch normalization + dropout actually help on a dataset this complex? It does, and the training curves make it pretty obvious.

---

## Dataset

**MLRSNet** by Xiaoman Qi et al.  
Source: [Kaggle — vigneshwar472/mlrs-net](https://www.kaggle.com/datasets/vigneshwar472/mlrs-net)

- 109,161 images at 256x256 px
- 46 scene categories (airplane, beach, forest, harbor, etc.)
- 60 semantic labels per image (multi-label, not multi-class)
- Pre-split into train / validation / test sets with label CSVs

---

## Model Architecture

Built a custom CNN with 4 convolutional blocks. Each block is two Conv→BN→ReLU layers followed by max pooling. Global Average Pooling at the end instead of flattening — this keeps the parameter count reasonable (~2M) and avoids memory issues.

```
Input (3 x 256 x 256)
  → Block 1: Conv(3→32) x2 + MaxPool   → 128x128
  → Block 2: Conv(32→64) x2 + MaxPool  →  64x64
  → Block 3: Conv(64→128) x2 + MaxPool →  32x32
  → Block 4: Conv(128→256) x2 + MaxPool→  16x16
  → Global Average Pool                →   1x1
  → FC(256→512) → ReLU → Dropout(0.5)
  → FC(512→60)  [raw logits]
```

Two variants trained for comparison:
- **With regularization**: Batch Norm after every conv/fc layer + Dropout(0.5) before classifier
- **Without regularization**: Same architecture, no BN, no Dropout

---

## Training Setup

| Setting | Value |
|---|---|
| Loss | BCEWithLogitsLoss (multi-label BCE) |
| Optimizer | Adam, lr=1e-3, weight decay=1e-4 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Early Stopping | patience=4 |
| Batch Size | 64 |
| Max Epochs | 15 |
| Hardware | Google Colab T4 GPU |

Augmentation on training set: random horizontal flip, random rotation (±10°), color jitter. Validation/test: normalization only.

---

## Results

### Model WITH Batch Normalization + Dropout (10 epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.1545 | 0.9401 | 0.1314 | 0.9483 |
| 5 | 0.0946 | 0.9621 | 0.0880 | 0.9643 |
| 10 | 0.0806 | 0.9677 | 0.0758 | 0.9692 |

### Model WITHOUT Regularization (5 epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.2076 | 0.9202 | 0.2046 | 0.9198 |
| 3 | 0.2011 | 0.9226 | 0.1852 | 0.9304 |
| 5 | 0.1612 | 0.9386 | 0.1506 | 0.9410 |

The regularized model converges faster and reaches lower loss — the gap is pretty visible even by epoch 5.

### Test Set Metrics (regularized model)

| Metric | Value |
|---|---|
| Macro F1 | 0.7214 |
| Micro F1 | 0.7976 |
| Macro Precision | 0.8884 |
| Macro Recall | 0.6343 |
| Best class | `containers` (F1 = 0.9478) |
| Worst class | `railway station` (F1 = 0.2868) |

The macro/micro F1 gap (~0.076) comes from class imbalance — the model handles frequent labels well but struggles on rare ones like `railway station` which don't appear often enough in training.

---

## Repo Structure

```
CNN_23-50531-1/
  CNN_23_50531_1.ipynb     ← full notebook with outputs
  best_with_reg.pth        ← saved weights (with BN + Dropout)
  best_without_reg.pth     ← saved weights (no regularization)
  (extracted figures and files)
  README.md
```

---

## How to Run

### On Google Colab (recommended)

1. Open `CNN_23_50531_1.ipynb` in Colab
2. Set runtime to T4 GPU: Runtime → Change runtime type → T4 GPU
3. Run Section 0 cells to mount Drive and download the dataset via Kaggle API
4. Run all remaining cells

### Kaggle API setup

Go to kaggle.com → Settings → API → Create New Token → download `kaggle.json` → upload when prompted in the notebook.

### Local setup

```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn tqdm pillow
```

Change `data_root` in the notebook to your local MLRSNet path and set `NUM_WORKERS = 0` on Windows.

---

## Key Observations

**Why regularization helps here**: MLRSNet has 60 output labels and a lot of label co-occurrence (e.g., `buildings` almost always appears with `road`). Without BN, the gradients in deeper layers become unstable and training is slower. Dropout forces the classifier head to not rely on specific neurons, which helps generalization across the 60-class output space.

**Best class (`containers`)**: Very distinctive visual pattern — rectangular uniform shapes, usually in port/shipping contexts. Easy for the model to pick up.

**Worst class (`railway station`)**: Visually similar to other infrastructure scenes (buildings + roads + vehicles). Low training support and high visual overlap with `intersection`, `parking_lot`, etc.

**Macro vs Micro F1**: The 0.076 gap tells you the model is doing well on common labels and worse on rare ones. Standard behavior for imbalanced multi-label problems — asymmetric loss or per-class threshold tuning would help.

---

## Dependencies

```
torch >= 2.0
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
Pillow
```

---

## References

- Qi, X. et al. "MLRSNet: A Multi-label High Spatial Resolution Remote Sensing Dataset for Semantic Image Interpretation." *Pattern Recognition*, 2020.
- Dataset on Kaggle: https://www.kaggle.com/datasets/vigneshwar472/mlrs-net
