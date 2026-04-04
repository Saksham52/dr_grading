import torch
import torch.nn as nn
import timm

class DRClassifier(nn.Module):
    def __init__(self, num_classes= 5, dropout=0.3):
        super(DRClassifier,self).__init__()
        
        #Load pretrained EfficientNet-B4
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=True,
            num_classes=0   #Remove original classifier
        )

        #Get the number of features coming out of backbone
        num_features = self.backbone.num_features

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    