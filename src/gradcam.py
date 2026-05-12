import torch
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        
        # Attach hooks to final conv layer
        self.model.backbone.conv_head.register_forward_hook(self._save_features)
        self.model.backbone.conv_head.register_full_backward_hook(self._save_gradients)
    
    def _save_features(self, module, input, output):
        self.feature_maps = output.detach()
    
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, image_tensor, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(image_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Calculate weights - average gradients across spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        
        # Weighted combination of feature maps
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize to 0-1
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx
    
    def overlay(self, cam, original_image):
        # Resize cam to match image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        return overlay