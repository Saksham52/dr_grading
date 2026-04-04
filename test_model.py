import torch
from src.model import DRClassifier

# Create model
model = DRClassifier(num_classes=5,dropout=0.3)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = model.to(device)

# Test with a fake batch
fake_input = torch.randn(4, 3, 224, 224).to(device)
output = model(fake_input)

print("Inpur shape: ", fake_input.shape)
print("Output shape:", output.shape)
print("Output:", output)