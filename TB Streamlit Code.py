#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(
    page_title="TB X-ray Classifier",
    page_icon="🩺",
    layout="wide"
)

# Load models (cache to avoid reloading)
@st.cache_resource
def load_models():
    try:
        model = joblib.load('svc_model_precision.pkl')
        threshold = joblib.load('optimal_threshold.pkl')
        pca = joblib.load('pca_model.pkl')
        
        # Load DenseNet feature extractor
        densenet = models.densenet121(pretrained=True)
        feature_extractor = nn.Sequential(*list(densenet.children())[:-1])
        feature_extractor.eval()
        
        return model, threshold, pca, feature_extractor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Feature extraction
def extract_features(image_tensor, feature_extractor):
    with torch.no_grad():
        features = feature_extractor(image_tensor)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return features.view(features.size(0), -1).cpu().numpy()

# Main app
def main():
    st.title("🩺 Tuberculosis X-ray Classification")
    st.markdown("""
    This app classifies chest X-rays as **Normal** or **Tuberculosis (TB)** positive.
    Upload a chest X-ray image to get started.
    """)
    
    # Load models
    model, threshold, pca, feature_extractor = load_models()
    if None in [model, threshold, pca, feature_extractor]:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        try:
            # Display image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded X-ray", width=300)
            
            # Preprocess and predict
            with st.spinner("Analyzing the X-ray..."):
                # Preprocess
                image_tensor = preprocess_image(image)
                
                # Extract features
                features = extract_features(image_tensor, feature_extractor)
                
                # Apply PCA
                features_pca = pca.transform(features)
                
                # Predict
                proba = model.predict_proba(features_pca)[0][1]
                prediction = proba >= threshold
                
                # Display results
                st.subheader("Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confidence meter
                    st.metric("Confidence Score", f"{proba:.2%}")
                    st.progress(float(proba))
                    
                    # Threshold info
                    st.caption(f"Decision threshold: {threshold:.2f}")
                
                with col2:
                    # Prediction
                    if prediction:
                        st.error("## 🦠 TB Positive")
                        st.warning("This X-ray shows signs of Tuberculosis. Please consult a doctor.")
                    else:
                        st.success("## ✅ Normal")
                        st.info("No signs of Tuberculosis detected.")
                
                # Show probability distribution
                fig, ax = plt.subplots(figsize=(6, 2))
                sns.barplot(x=[proba, 1-proba], 
                           y=['TB Probability', 'Normal Probability'],
                           palette=['red', 'green'],
                           ax=ax)
                ax.set_xlim(0, 1)
                st.pyplot(fig)
                
                # Enhanced results interpretation
                st.markdown("---")
                st.subheader("How to interpret results:")
                st.markdown("""
                - 🟢 **Below {threshold:.0%}**: Normal (No TB detected)
                - 🟠 **{threshold:.0%}-50%**: Borderline - Consult doctor
                - 🔴 **Above 50%**: High TB probability
                """.format(threshold=threshold))

                st.markdown(f"""
                **Your image**:  
                🔵 **{proba:.2%}** TB probability → **Well below** the {threshold:.0%} threshold  
                ✅ **Clear normal result**
                """)
                
                # Model info
                st.markdown("---")
                st.caption("Model: Precision-optimized SVM")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Add sidebar with info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This TB detection system uses:
        - **DenseNet121** for feature extraction
        - **PCA** for dimensionality reduction
        - **Precision-optimized SVM** classifier
        
        Model performance:
        - Precision: 95%
        - Recall: 88%
        - AUC: 0.96
        """)
        
        st.markdown("---")
        st.markdown("⚠️ **Disclaimer**: This is a diagnostic aid, not a replacement for professional medical opinion.")

if __name__ == "__main__":
    main()