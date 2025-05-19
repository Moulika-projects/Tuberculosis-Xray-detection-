#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data_path = r"C:\Users\Asus\Downloads\TB_Chest_Radiography_Database" 
task_type = "classification"


# In[3]:


def detect_data_type(path):
    if path.endswith((".csv", ".xlsx", ".json")):
        return "structured"
    else:
        return "image"

data_type = detect_data_type(data_path)
print("📦 Detected Data Type:", data_type)


# In[10]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
import joblib

# Initialize placeholders
X = None
y = None
df = None
dataset = None
class_names = []

# Placeholder for data path and type (make sure these are set before running the code)
data_path = r"C:\Users\Asus\Downloads\TB_Chest_Radiography_Database"  # Update this with your actual path
data_type = "image"  # Or "structured", depending on your dataset

# Handle structured data (CSV or Excel)
if data_type == "structured":
    print("🔧 Preprocessing structured data...")

    # Load dataset
    df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_excel(data_path)
    print("Data Shape:", df.shape)
    display(df.head())  # Shows the first few rows of the data

    # Target column input (customize as needed)
    target_col = input("Enter the name of the target column: ")

    # Split features and target
    X_raw = df.drop(columns=[target_col])
    y = df[target_col]

    # Column types
    num_cols = X_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()

    # Pipelines for numerical and categorical features
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine pipelines using ColumnTransformer
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # Apply transformations
    X = full_pipeline.fit_transform(X_raw)
    print("✅ Structured data preprocessed.")

# Handle image data (X-ray images)
elif data_type == "image":
    print("🖼️ Preprocessing image dataset...")

    # Image transformation pipeline (resize, normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Simple normalization
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(data_path, transform=transform)
    class_names = dataset.classes

    # Get all labels
    labels = np.array([label for _, label in dataset.imgs])
    
    # Get indices of TB and Normal images
    tb_indices = np.where(labels == 1)[0]  # TB class (1)
    normal_indices = np.where(labels == 0)[0]  # Normal class (0)

    # Randomly sample 700 from each class
    sample_tb = np.random.choice(tb_indices, 700, replace=False)
    sample_normal = np.random.choice(normal_indices, 700, replace=False)

    # Combine and shuffle indices
    balanced_indices = np.concatenate([sample_tb, sample_normal])
    np.random.shuffle(balanced_indices)

    # Create balanced dataset using Subset
    balanced_dataset = Subset(dataset, balanced_indices)

    # Verify class distribution
    balanced_labels = np.array([label for _, label in balanced_dataset])
    print("\nFinal Class Distribution:")
    print(f"{class_names[0]}: {sum(balanced_labels == 0)}")
    print(f"{class_names[1]}: {sum(balanced_labels == 1)}")
    print(f"✅ Loaded {len(balanced_dataset)} images across {len(class_names)} classes.\n")

    # Visualize samples from both classes
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle("Sample Images from Each Class", fontsize=16)
    
    for i in range(3):
        # Normal samples
        idx = np.where(balanced_labels == 0)[0][i]
        img, _ = balanced_dataset[idx]
        ax[0,i].imshow(img.permute(1,2,0).numpy() * 0.5 + 0.5)
        ax[0,i].set_title(class_names[0])
        ax[0,i].axis('off')
        
        # TB samples
        idx = np.where(balanced_labels == 1)[0][i]
        img, _ = balanced_dataset[idx]
        ax[1,i].imshow(img.permute(1,2,0).numpy() * 0.5 + 0.5)
        ax[1,i].set_title(class_names[1])
        ax[1,i].axis('off')
    
    plt.tight_layout()
    plt.show()

else:
    print("⚠️ Unsupported or unknown data format.")


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import cv2
from torchvision.utils import make_grid

def run_eda(data_type, df=None, dataset=None, y=None, class_names=None):
    if data_type == "structured":
        print("📊 Running EDA for structured data...")
        display(df.describe())

        # Missing value heatmap
        plt.figure(figsize=(10, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Value Heatmap")
        plt.show()

        # Target distribution
        if y is not None and y.nunique() < 20:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=y)
            plt.title("Target Variable Distribution")
            plt.xticks(rotation=45)
            plt.show()

        # Histograms
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        df[num_cols].hist(figsize=(12, 8), bins=20)
        plt.suptitle("Histograms of Numeric Features")
        plt.show()

        # Correlation heatmap
        if len(num_cols) >= 2:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.show()
             # Boxplots for first few numeric columns
        if y is not None and y.nunique() <= 5:
            for col in num_cols[:4]:
                plt.figure(figsize=(6, 4))
                sns.boxplot(x=y, y=df[col])
                plt.title(f"{col} by Target")
                plt.show()

    elif data_type == "image":
        print("🖼️ Running EDA for image data...")

        # Class distribution bar plot
        label_list = [label for _, label in dataset.imgs]
        label_names = [class_names[label] for label in label_list]
        sns.countplot(x=label_names)
        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=45)
        plt.show()
        
         # Show sample images
        loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
        imgs, labels = next(iter(loader))
        grid = make_grid(imgs, nrow=3)
        np_grid = grid.permute(1, 2, 0).numpy()
        plt.figure(figsize=(8, 5))
        plt.imshow(np_grid * 0.5 + 0.5)  # unnormalize
        plt.title("Sample Images")
        plt.axis('off')
        plt.show()

        # Average image per class
        for class_idx, class_name in enumerate(class_names):
            class_imgs = [img for img, label in dataset if label == class_idx]
            if len(class_imgs) > 0:
                avg_img = torch.stack(class_imgs).mean(dim=0)
                plt.figure(figsize=(4, 4))
                plt.imshow(avg_img.permute(1, 2, 0).numpy() * 0.5 + 0.5)
                plt.title(f"Average Image - {class_name}")
                plt.axis('off')
                plt.show()
    else:
        print("⚠️ EDA not supported for this data type.")


# In[12]:


from torch.utils.data import Subset, DataLoader
import numpy as np

# Set seed
np.random.seed(42)

# Extract labels
labels = np.array([label for _, label in dataset.imgs])
pos_indices = np.where(labels == 1)[0]
neg_indices = np.where(labels == 0)[0]

# Sample balanced indices
min_class_size = min(len(pos_indices), len(neg_indices), 700)
sampled_pos = np.random.choice(pos_indices, min_class_size, replace=False)
sampled_neg = np.random.choice(neg_indices, min_class_size, replace=False)

# Combine and shuffle
balanced_indices = np.concatenate([sampled_pos, sampled_neg])
np.random.shuffle(balanced_indices)

# Create balanced dataset and loader
balanced_dataset = Subset(dataset, balanced_indices)
loader = DataLoader(balanced_dataset, batch_size=32, shuffle=True)


# In[13]:


run_eda(data_type, df=df, dataset=dataset, y=y, class_names=class_names) 


# In[14]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import DataLoader

# Placeholder
X_features = None

if data_type == "structured":
    print("🔧 Feature engineering for structured data...")

    # Polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    print(f"Original shape: {X.shape}")
    print(f"Expanded shape: {X_poly.shape}")

    # Feature selection using ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=min(50, X_poly.shape[1]))
    X_features = selector.fit_transform(X_poly, y)

    print(f"Selected top {X_features.shape[1]} features.")

elif data_type == "image":
    print("🖼️ Enhanced feature extraction for image data using DenseNet...")

    # Load pretrained DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    densenet = models.densenet121(pretrained=True)
    feature_extractor = nn.Sequential(*list(densenet.children())[:-1])  # Remove final FC layer
    feature_extractor.to(device).eval()

    # Add adaptive pooling to standardize feature size
    adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    # Prepare DataLoader
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features = []
    labels = []

    # Extract features
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting CNN features"):
            imgs = imgs.to(device)
            outputs = feature_extractor(imgs)
            outputs = adaptive_pool(outputs)  # Apply adaptive pooling
            outputs = outputs.view(outputs.size(0), -1)  # Flatten
            features.append(outputs.cpu().numpy())
            labels.extend(lbls.numpy())

    X_features = np.concatenate(features, axis=0)
    y = np.array(labels)

    print(f"Extracted DenseNet features shape: {X_features.shape}")

    # Apply PCA
    pca = PCA(n_components=100, random_state=42)
    X_features = pca.fit_transform(X_features)
    
    import joblib
    joblib.dump(pca, "pca_model.pkl")
    print("✅ PCA object saved as 'pca_model.pkl'")

    print(f"Reduced feature size after PCA: {X_features.shape}")


# In[15]:


from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
import numpy as np

def feature_selection(X, y=None, method="anova", k=50):
    """
    Adaptive feature selection based on the specified method.

    Parameters:
        X (array-like): Feature matrix.
        y (array-like): Target vector (required for supervised methods).
        method (str): Selection method: 'anova', 'mutual_info', 'variance', 'tree', or 'lasso'.
        k (int): Number of top features to select (for applicable methods).

    Returns:
        X_new (array-like): Feature matrix after selection.
    """
    
    if method == "anova":
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        print(f"✅ Selected {X_new.shape[1]} features using ANOVA.")
        return X_new

    elif method == "mutual_info":
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_new = selector.fit_transform(X, y)
        print(f"✅ Selected {X_new.shape[1]} features using Mutual Information.")
        return X_new

    elif method == "variance":
        selector = VarianceThreshold(threshold=0.01)
        X_new = selector.fit_transform(X)
        print(f"✅ Selected {X_new.shape[1]} features using Variance Threshold.")
        return X_new

    elif method == "tree":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        importance = model.feature_importances_
        top_k_idx = importance.argsort()[-k:]
        X_new = X[:, top_k_idx]


    elif method == "lasso":
        lasso = Lasso(alpha=0.01)
        lasso.fit(X, y)
        importance = np.abs(lasso.coef_)
        top_k_idx = importance.argsort()[-k:]
        X_new = X[:, top_k_idx]
        print(f"✅ Selected {X_new.shape[1]} features using Lasso.")
        return X_new

    else:
        raise ValueError("❌ Unsupported feature selection method. Choose from 'anova', 'mutual_info', 'variance', 'tree', or 'lasso'.")


# In[16]:


#Skip feature selection — use PCA output directly
X_selected = X_features  # X_features is the PCA-reduced feature matrix

# Print shape for confirmation
print(f"✅ Using PCA output directly. Shape: {X_selected.shape}")


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (roc_curve, auc, RocCurveDisplay, classification_report,
                             mean_squared_error, silhouette_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate(X, y=None, task_type="classification", algorithm="rf"):
    if task_type == "classification":
        print("🎯 Classification Mode")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if algorithm == "rf":
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
            param_dist = {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        elif algorithm == "svm":
            model = SVC(probability=True, class_weight='balanced')
            param_dist = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'],
                'kernel': ['linear', 'rbf', 'poly']
            }
        else:
            raise ValueError("Unsupported classification algorithm.")

        random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                           n_iter=20, cv=3, scoring='accuracy',
                                           n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)

        print(f"✅ Best Parameters: {random_search.best_params_}")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    elif task_type == "regression":
        print("📈 Regression Mode")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if algorithm == "ridge":
            model = Ridge()
            param_grid = {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr']
            }
        elif algorithm == "linear":
            model = LinearRegression()
            param_grid = {}  # No important hyperparams
        elif algorithm == "rfr":
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        else:
            raise ValueError("Unsupported regression algorithm.")

        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print(f"✅ Best Parameters: {grid.best_params_}")
        print(f"📉 Mean Squared Error: {mse:.4f}")

    elif task_type == "clustering":
        print("🔵 Clustering Mode")

        if algorithm == "kmeans":
            model = KMeans(random_state=42, n_init='auto')
            param_grid = {
                'n_clusters': [2, 3, 4, 5],
                'init': ['k-means++', 'random'],
                'n_init': [10, 20]
            }
        elif algorithm == "dbscan":
            model = DBSCAN()
            param_grid = {
                'eps': [0.3, 0.5, 0.7],
                'min_samples': [5, 10]
            }
        elif algorithm == "agglo":
            model = AgglomerativeClustering()
            param_grid = {
                'n_clusters': [2, 3, 4],
                'linkage': ['ward', 'complete', 'average']
            }
        else:
            raise ValueError("Unsupported clustering algorithm.")

        model.fit(X)
        cluster_labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
        silhouette = silhouette_score(X, cluster_labels)
        print(f"✅ Silhouette Score: {silhouette:.4f}")

    else:
        print("⚠️ Unsupported task type.")


# In[18]:


# Replace the In[28] cell with this:

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# Define parameter grid for SVM optimized for precision
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf'],
    'class_weight': [{0: 1, 1: 3}]  # Higher weight for TB class to reduce false negatives
}

# Use precision as scoring metric to reduce false positives
grid = GridSearchCV(
    SVC(probability=True),
    param_grid,
    cv=5,
    scoring='precision',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

# Get best model
best_svc = grid.best_estimator_
print(f"Best parameters: {grid.best_params_}")

# Evaluate
y_pred = best_svc.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Find optimal threshold for high precision
y_probs = best_svc.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
optimal_idx = np.argmax(precision >= 0.95)  # Target 95% precision
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold for 95% precision: {optimal_threshold:.4f}")

# Save model and threshold
joblib.dump(best_svc, 'svc_model_precision.pkl')
joblib.dump(optimal_threshold, 'optimal_threshold.pkl')
print("✅ Model and threshold saved")


# In[19]:


# Load model and threshold
model = joblib.load('svc_model_precision.pkl')
threshold = joblib.load('optimal_threshold.pkl')

# Make predictions with threshold
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= threshold).astype(int)

# Evaluate
print("Classification Report with Optimized Threshold:")
print(classification_report(y_test, y_pred))

# New confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix with Optimized Threshold")
plt.show()


# In[ ]:




