import cv2
import matplotlib.pyplot as plt

raw = cv2.imread('data/raw/train_images/000c1434d8d7.png')
raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

processed = cv2.imread('data/processed/000c1434d8d7.png')
processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(raw)
axes[0].set_title('Raw')
axes[1].imshow(processed)
axes[1].set_title('Processed')
plt.savefig('outputs/plots/preprocess_check.png')
print("Saved to outputs/plots/preprocess_check.png")