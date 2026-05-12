# Diabetic Retinopathy Grading System

An automated deep learning system that classifies diabetic retinopathy severity from retinal fundus images using EfficientNet-B4 with Grad-CAM explainability.

## Project Structure

```
dr_grading/
├── src/
│   ├── preprocess.py     # Ben Graham preprocessing pipeline
│   ├── dataset.py        # PyTorch Dataset class with weighted sampling
│   ├── model.py          # EfficientNet-B4 with custom classification head
│   ├── train.py          # Two-phase training loop with Focal Loss
│   ├── evaluate.py       # Kappa score, confusion matrix, ROC curves
│   ├── gradcam.py        # Grad-CAM heatmap implementation
│   └── __init__.py
├── outputs/
│   ├── checkpoints/      # Saved model weights
│   ├── plots/            # Confusion matrix, ROC curves, preprocessing comparison
│   └── gradcam/          # Sample heatmap outputs
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
└── README.md
```

## Dataset

APTOS 2019 Blindness Detection - 3,662 labeled retinal fundus images across 5 severity grades.

| Grade | Label | Count |
|-------|-------|-------|
| 0 | No DR | 1805 |
| 1 | Mild | 370 |
| 2 | Moderate | 999 |
| 3 | Severe | 193 |
| 4 | Proliferative | 295 |

Download from: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

Once downloaded, place the files so the structure looks like this:
```
data/raw/
├── train_images/
└── train.csv
```

## Setup

The model checkpoint is included in this zip at `outputs/checkpoints/best_model.pth`. You do not need to train the model - just set up the environment and run the app.

**1 - Create a virtual environment:**
```bash
python -m venv dr_grading_env
```

**2 - Activate it:**

On Windows:
```bash
dr_grading_env\Scripts\activate
```

On Mac/Linux:
```bash
source dr_grading_env/bin/activate
```

**3 - Install PyTorch with CUDA (run this before requirements.txt):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you do not have a CUDA-capable GPU, install the CPU version instead:
```bash
pip install torch torchvision torchaudio
```

**4 - Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

**5 - Run the web app:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Upload any retinal fundus image to get a severity grade and Grad-CAM heatmap.

## Every Time You Open a New Terminal

You need to activate the virtual environment before running anything:

On Windows:
```bash
dr_grading_env\Scripts\activate
```

On Mac/Linux:
```bash
source dr_grading_env/bin/activate
```

## Optional - Preprocess and Retrain From Scratch

If you want to reproduce the full training pipeline from scratch, download the dataset first, then:

```bash
# Preprocess all 3,662 images
python src/preprocess.py

# Train the model (takes 2-3 hours on a GPU)
python -m src.train

# Evaluate on test set
python -m src.evaluate
```

## What Ben Graham Preprocessing Does

Raw fundus images have three problems that hurt model performance. Ben Graham preprocessing fixes all three:

1. Removes black borders around the circular retinal area using contour detection
2. Subtracts a Gaussian-blurred version of the image to fix uneven lighting across the retina
3. Enhances local contrast so fine structures like microaneurysms and blood vessels are more visible
4. Resizes to 380x380 to match EfficientNet-B4's design specification

## Key Design Decisions

**Why EfficientNet-B4?**
Pretrained on 1.2 million ImageNet images - it already knows how to detect edges, curves, and textures. Fine-tuning it for retinal images is far more effective than training from scratch on only 3,662 images. It was also specifically designed for 380x380 input, which captures fine retinal detail that lower resolutions miss.

**Why Focal Loss?**
The dataset has 1,805 healthy images and only 193 Severe images. Standard loss functions get dominated by the easy majority cases. Focal Loss reduces the learning signal from examples the model already handles well, so rare critical classes actually drive the training.

**Why Grad-CAM?**
A grade alone is not enough for clinical use. Doctors need to see what the model was looking at to trust and verify the decision. Grad-CAM generates heatmaps showing which retinal regions influenced the prediction, and also helps catch cases where the model might be relying on image artifacts instead of actual pathology.

**Why two-phase training?**
The classification head starts with random weights. Training the full model immediately would destroy EfficientNet's pretrained knowledge through large random gradients. Phase 1 freezes the backbone for 5 epochs while the head stabilizes. Phase 2 unfreezes everything and fine-tunes at a much lower learning rate to gently adapt the backbone to retinal images.

## Results

| Metric | Value |
|--------|-------|
| Best Validation Kappa | 0.8543 |
| No DR AUC | 0.99 |
| Severe AUC | 0.93 |
| Moderate AUC | 0.90 |
| Mild AUC | 0.89 |
| Proliferative AUC | 0.89 |

The model never predicts healthy (Grade 0) for a Severe or Proliferative case. The clinically dangerous false negatives do not occur in the test set.

## Evaluation Metric

Quadratic Weighted Kappa (QWK) is the primary metric, the same one used in the original APTOS 2019 Kaggle competition. Unlike accuracy, it accounts for the ordinal nature of the grades and penalizes predictions that are far from the true grade more heavily than close ones. A score above 0.8 is considered almost perfect agreement.
