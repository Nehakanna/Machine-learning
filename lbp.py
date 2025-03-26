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

from skimage.feature import local_binary_pattern
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess_image(image, size=(64, 64)):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, size)
    image = image.astype("float32") / 255.0
    return image

def extract_lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return hist.astype("float32") / hist.sum()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

# Extract LBP features
x_train_features = np.array([extract_lbp_features(preprocess_image(img)) for img in x_train[:1000]])
x_test_features = np.array([extract_lbp_features(preprocess_image(img)) for img in x_test[:200]])

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

print("Feature Extraction Method: LBP")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
