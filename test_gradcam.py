import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.model import DRClassifier
from src.gradcam import GradCAM
from src.dataset import val_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = DRClassifier(num_classes=5, dropout=0.3).to(device)
model.load_state_dict(torch.load(
    'outputs/checkpoints/best_model.pth',
    map_location=device,
    weights_only=True
))

# Load a sample image
img_path = 'data/processed/000c1434d8d7.png'
original = cv2.imread(img_path)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Preprocess for model
tensor = val_transform(original).unsqueeze(0).to(device)

# Generate Grad-CAM
gradcam = GradCAM(model)
cam, predicted_class = gradcam.generate(tensor)
overlay = gradcam.overlay(cam, original)

# Grade names
grades = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
print(f"Predicted class: {grades[predicted_class]}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(original)
axes[0].set_title('Preprocessed Image')
axes[1].imshow(overlay)
axes[1].set_title(f'Grad-CAM - Predicted: {grades[predicted_class]}')
plt.savefig('outputs/gradcam/sample_gradcam.png')
print("Saved to outputs/gradcam/sample_gradcam.png")