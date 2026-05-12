import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from src.model import DRClassifier
from src.dataset import get_dataloaders

def evaluate(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    _, _, test_loader = get_dataloaders(
        csv_path=config['csv_path'],
        img_dir=config['img_dir'],
        batch_size=config['batch_size']
    )
    
    # Load model
    model = DRClassifier(num_classes=5, dropout=0.3).to(device)
    model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device, weights_only=True))    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Quadratic Weighted Kappa
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    print(f"Test Quadratic Weighted Kappa: {kappa:.4f}")
    
    return all_labels, all_preds, all_probs


def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_roc_curves(labels, probs, save_path):
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    n_classes = 5
    
    # Binarize labels
    labels_bin = label_binarize(labels, classes=[0, 1, 2, 3, 4])
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves per Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curves saved to {save_path}")

if __name__ == "__main__":
    config = {
        'csv_path': 'data/raw/train.csv',
        'img_dir': 'data/processed',
        'batch_size': 16,
        'checkpoint_path': 'outputs/checkpoints/best_model.pth'
    }
    
    labels, preds, probs = evaluate(config)
    
    plot_confusion_matrix(
        labels, preds,
        save_path='outputs/plots/confusion_matrix.png'
    )
    
    plot_roc_curves(
        labels, probs,
        save_path='outputs/plots/roc_curves.png'
    )