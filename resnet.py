import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.datasets import cifar10

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
\

resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
resnet_extractor = Model(inputs=resnet_model.input, outputs=resnet_model.output)

def preprocess_image(image, size=(64, 64)):
    image = cv2.resize(image, size)  
    
    if image.shape[-1] == 1: 
        image = np.concatenate([image] * 3, axis=-1)
    
    image = preprocess_input(image)  
    return image

def extract_resnet_features(image):
    image = np.expand_dims(image, axis=0)  
    features = resnet_extractor.predict(image)
    return features.flatten()  
    
# Extract ResNet features
x_train_features = np.array([extract_resnet_features(preprocess_image(img)) for img in x_train[:1000]])
x_test_features = np.array([extract_resnet_features(preprocess_image(img)) for img in x_test[:200]])

# Standardize features
scaler = StandardScaler()
x_train_features = scaler.fit_transform(x_train_features)
x_test_features = scaler.transform(x_test_features)

# Train classifier (Decision Tree)
classifier = DecisionTreeClassifier()
classifier.fit(x_train_features, y_train[:1000])
y_pred = classifier.predict(x_test_features)

# Evaluate performance
accuracy = accuracy_score(y_test[:200], y_pred)
precision = precision_score(y_test[:200], y_pred, average='macro')
recall = recall_score(y_test[:200], y_pred, average='macro')
f1 = f1_score(y_test[:200], y_pred, average='macro')

print("Feature Extraction Method: ResNet50")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
