import os
import cv2 
import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DRDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform = None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        
        #Get filename and label
        row = self.dataframe.iloc[idx]
        img_id = row['id_code']
        label = row['diagnosis']

        # Build Image full path
        image_path = os.path.join(self.img_dir, img_id+'.png')

        # Load Image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Apply transform
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype = torch.long)


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean= [0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225])
])

def get_dataloaders(csv_path, img_dir, batch_size= 16):
    #Load CSV
    df = pd.read_csv(csv_path)

    #Stratified split
    from sklearn.model_selection import train_test_split

    train_df , temp_df = train_test_split(
        df,
        test_size= 0.25,
        stratify= df['diagnosis'],
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.4,
        stratify=temp_df['diagnosis'],
        random_state=42
    )

    #Create datasets
    train_dataset = DRDataset(train_df, img_dir, transform=train_transform)
    val_dataset = DRDataset(val_df, img_dir, transform=val_transform)
    test_dataset = DRDataset(test_df, img_dir, transform=val_transform)
 
    #Weighted sampling for class imbalance
    class_counts = df['diagnosis'].value_counts().sort_index().values
    class_weights = 1.0/ class_counts
    sample_weights = [class_weights[label] for label in train_df['diagnosis'].values]
    sample_weights = torch.tensor(sample_weights,dtype=torch.float)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )
    #  Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size= batch_size, sampler = sampler)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader








