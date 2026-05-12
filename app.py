import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from src.model import DRClassifier
from src.gradcam import GradCAM
from src.dataset import val_transform
from src.preprocess import preprocess_image
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Diabetic Retinopathy Grading",
    page_icon="👁",
    layout="wide"
)

# Grade info
GRADES = {
    0: ("No DR", "green", "No signs of diabetic retinopathy detected."),
    1: ("Mild", "yellow", "Microaneurysms only."),
    2: ("Moderate", "orange", "More than microaneurysms but less than severe DR."),
    3: ("Severe", "red", "Severe DR. Refer to specialist immediately."),
    4: ("Proliferative", "darkred", "Proliferative DR. Urgent specialist referral required.")
}

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DRClassifier(num_classes=5, dropout=0.3).to(device)
    model.load_state_dict(torch.load(
        'outputs/checkpoints/best_model.pth',
        map_location=device,
        weights_only=True
    ))
    return model, device

def predict(image_path, model, device):
    # Preprocess
    img = preprocess_image(image_path)
    
    # Convert for display
    display_img = img.copy()
    
    # Prepare tensor
    tensor = val_transform(img).unsqueeze(0).to(device)
    
    # Grad-CAM
    gradcam = GradCAM(model)
    cam, predicted_class = gradcam.generate(tensor)
    overlay = gradcam.overlay(cam, display_img)
    
    # Get probabilities
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    
    return predicted_class, probs, display_img, overlay

# UI
st.title("👁 Diabetic Retinopathy Grading System")
st.markdown("Upload a retinal fundus image to get an automated DR severity grade with visual explanation.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose a fundus image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            f.write(uploaded_file.read())
            temp_path = f.name
        
        # Load and display original
        image = Image.open(temp_path)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    if uploaded_file is not None:
        with st.spinner("Analyzing image..."):
            model, device = load_model()
            predicted_class, probs, processed_img, overlay = predict(temp_path, model, device)
        
        grade_name, color, description = GRADES[predicted_class]
        
        st.subheader("Results")
        st.markdown(f"### Predicted Grade: {predicted_class} - {grade_name}")
        st.markdown(f"**{description}**")
        
        # Probability bars
        st.subheader("Confidence per Grade")
        for i, (name, _, _) in GRADES.items():
            st.progress(float(probs[i]), text=f"{name}: {probs[i]*100:.1f}%")
        
        # Grad-CAM
        st.subheader("Grad-CAM Explanation")
        st.image(overlay, caption="Regions the model focused on", use_container_width=True)
        
        # Cleanup
        os.unlink(temp_path)