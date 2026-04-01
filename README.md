# Diabetic Retinopathy Grading System

An automated deep learning system that classifies diabetic retinopathy severity from retinal fundus images using EfficientNet-B4 with Grad-CAM explainability.

## Project Structure
```
dr_grading/
├── data/
│   ├── raw/              # Original APTOS 2019 images (not tracked)
│   └── processed/        # Ben Graham preprocessed images (not tracked)
├── src/
│   ├── preprocess.py     # Ben Graham preprocessing pipeline ✅ DONE
│   ├── dataset.py        # PyTorch Dataset class 🔄 IN PROGRESS
│   ├── model.py          # EfficientNet-B4 + classification head
│   ├── train.py          # Training loop with Focal Loss
│   ├── evaluate.py       # Metrics, confusion matrix, ROC curves
│   └── gradcam.py        # Grad-CAM implementation
├── outputs/
│   ├── checkpoints/      # Saved model weights (not tracked)
│   ├── plots/            # Training curves, confusion matrix
│   └── gradcam/          # Heatmap outputs
├── app.py                # Streamlit demo
└── requirements.txt      # Dependencies
```

## Dataset
APTOS 2019 Blindness Detection — 3,662 labeled retinal fundus images across 5 severity grades:

| Grade | Label | Count |
|-------|-------|-------|
| 0 | No DR | 1805 |
| 1 | Mild | 370 |
| 2 | Moderate | 999 |
| 3 | Severe | 193 |
| 4 | Proliferative | 295 |

Download from: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

Place in `data/raw/` — folder structure should be:
```
data/raw/
├── train_images/
└── train.csv
```

## Setup

**1 — Create conda environment:**
```bash
conda create -n dr_grading python=3.10
conda activate dr_grading
```

**2 — Install PyTorch with CUDA:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3 — Install dependencies:**
```bash
pip install timm opencv-python Pillow numpy pandas scikit-learn matplotlib seaborn albumentations grad-cam streamlit jupyter
```

**4 — Preprocess images:**
```bash
python src/preprocess.py
```
This applies Ben Graham preprocessing to all 3,662 images and saves them to `data/processed/`.

## What Ben Graham Preprocessing Does
1. Removes black borders around the circular retinal area
2. Subtracts a Gaussian-blurred version of the image to fix uneven lighting
3. Enhances local contrast so fine details like microaneurysms are more visible
4. Resizes to 224×224

## Progress

### ✅ Completed
- Environment setup (Python 3.10, PyTorch 2.5.1 + CUDA 12.1)
- Dataset downloaded and verified (3,662 images)
- Ben Graham preprocessing pipeline (`src/preprocess.py`)
- All 3,662 images preprocessed and saved to `data/processed/`

### 🔄 In Progress
- PyTorch Dataset class (`src/dataset.py`)
  - Load images and labels from CSV
  - Stratified train/val/test split
  - Weighted sampling for class imbalance
  - Data augmentation

### 📋 Next Steps
1. `src/dataset.py` — Dataset class with weighted sampling
2. `src/model.py` — EfficientNet-B4 with custom classification head
3. `src/train.py` — Two phase training with Focal Loss
4. `src/evaluate.py` — Quadratic Weighted Kappa, confusion matrix, ROC curves
5. `src/gradcam.py` — Grad-CAM heatmap generation
6. `app.py` — Streamlit demo

## Key Concepts

**Why EfficientNet-B4?**
Pretrained on ImageNet — already knows edges, curves, textures. We fine-tune it for retinal images instead of training from scratch on only 3,662 images.

**Why Focal Loss?**
Dataset is heavily imbalanced — 1,805 healthy cases vs 193 Severe cases. Focal Loss silences easy majority cases so the model focuses on learning hard minority cases.

**Why Grad-CAM?**
Doctors need to see where the model looked, not just what it predicted. Grad-CAM generates heatmaps showing which retinal regions influenced the decision.

## Evaluation Metric
Quadratic Weighted Kappa — penalizes predictions that are far from the true grade more heavily than close ones. More appropriate than accuracy for ordinal medical grading.

## Target Performance
Quadratic Weighted Kappa ≥ 0.85 on held-out test set.