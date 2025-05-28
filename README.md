🏥 AI-Powered Tuberculosis Detection from Chest X-rays

🔍 Overview
 This project focuses on the automated detection of Tuberculosis (TB) from chest X-ray images using deep learning and machine learning techniques. Leveraging a publicly available dataset of radiographic images classified as either Normal or Tuberculosis, the system uses a pretrained ResNet50 Convolutional Neural Network (CNN) to extract deep features from each image. These high-dimensional feature vectors are then reduced using Principal Component Analysis (PCA) to optimize performance and minimize computational load. A Support Vector Machine (SVM) classifier is trained on the reduced features to distinguish between normal and TB-infected lungs. The entire pipeline is integrated into a user-friendly Streamlit web application, allowing users to upload X-ray images and receive instant predictions along with confidence scores. This model provides a lightweight, deployable solution to assist radiologists and healthcare professionals in early TB diagnosis, especially in low-resource settings.

- DenseNet121 CNN for feature extraction
- PCA (100 components) for dimensionality reduction  
- Precision-tuned SVM (12% decision threshold)
- Balanced dataset (700 Normal + 700 TB images)


#Install dependencies 
pip install -r requirements.txt
Machine Learning  &           Deep Learning
torch==2.2.1                  PyTorch (DenseNet121 backbone)
torchvision==0.17.1           Image preprocessing/datasets
scikit-learn==1.4.0           SVM, PCA, metrics

Data Handling
numpy==1.26.0                 Array operations
pandas==2.2.0                 Dataframes (if structured data used)
joblib==1.3.2                 Model serialization

Image Processing
pillow==10.1.0                Image loading
opencv-python==4.8.1          Advanced image ops (if needed)

Visualization
matplotlib==3.8.3             Plots/EDA
seaborn==0.13.0               Enhanced visualizations

App Framework
streamlit==1.32.0             Web interface

Utilities
tqdm==4.66.1                  Progress bars

# Why Each Dependency Matters

torch & torchvision

Essential for DenseNet121 feature extraction
Handles image transformations (Resize(), Normalize())

scikit-learn

Powers SVM classifier (SVC())
Manages PCA (PCA()) and metrics (precision_recall_curve)

streamlit

Turns your Python script into a web app
Generates UI for image uploads/results display

joblib

Saves/Loads trained models (.pkl files)
Critical for svc_model_precision.pkl, pca_model.pkl