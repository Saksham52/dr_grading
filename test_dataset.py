from src.dataset import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    csv_path='data/raw/train.csv',
    img_dir='data/processed',
    batch_size=16
)

# Get one batch
images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch labels:", labels)
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
print("Test batches:", len(test_loader))