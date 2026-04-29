import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import cohen_kappa_score
from src.model import DRClassifier
from src.dataset import get_dataloaders


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss
    
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return avg_loss, kappa

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return avg_loss, kappa

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=config['csv_path'],
        img_dir=config['img_dir'],
        batch_size=config['batch_size']
    )

    # Create model
    model = DRClassifier(num_classes=5, dropout=0.3).to(device)

    # Class weights for Focal Loss
    class_counts = np.array([1805, 370, 999, 193, 295])
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
    criterion = FocalLoss(gamma=2, weight=class_weights)

    # Phase 1 - frozen backbone
    print("\nPhase 1 - Training classification head only")
    model.freeze_backbone()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['phase1_lr']
    )

    best_kappa = 0
    for epoch in range(config['phase1_epochs']):
        train_loss, train_kappa = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_kappa = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config['phase1_epochs']} "
              f"Train Loss: {train_loss:.4f} Train Kappa: {train_kappa:.4f} "
              f"Val Loss: {val_loss:.4f} Val Kappa: {val_kappa:.4f}")

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), config['checkpoint_path'])
            print(f"Saved best model with kappa: {best_kappa:.4f}")

    # Phase 2 - unfreeze backbone
    print("\nPhase 2 - Fine tuning full model")
    model.unfreeze_backbone()
    optimizer = optim.Adam(model.parameters(), lr=config['phase2_lr'])

    for epoch in range(config['phase2_epochs']):
        train_loss, train_kappa = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_kappa = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config['phase2_epochs']} "
              f"Train Loss: {train_loss:.4f} Train Kappa: {train_kappa:.4f} "
              f"Val Loss: {val_loss:.4f} Val Kappa: {val_kappa:.4f}")

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), config['checkpoint_path'])
            print(f"Saved best model with kappa: {best_kappa:.4f}")

    print(f"\nTraining complete. Best validation kappa: {best_kappa:.4f}")
    return model

if __name__ == "__main__":
    config = {
    'csv_path': 'data/raw/train.csv',
    'img_dir': 'data/processed',
    'batch_size': 8,
    'phase1_lr': 1e-3,
    'phase1_epochs': 5,
    'phase2_lr': 1e-5,
    'phase2_epochs': 25,
    'checkpoint_path': 'outputs/checkpoints/best_model.pth'
    }
    
    train(config)