# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from google.colab import drive
drive.mount('/content/drive')

train_data = pd.read_csv("/content/drive/My Drive/Machine Learning/sign_mnist_train.csv")
train_data.head()

test_data = pd.read_csv("/content/drive/My Drive/Machine Learning/sign_mnist_test.csv")
test_data.head()

X_train = train_data.iloc[:, 1:]  # Features (pixel values)
y_train = train_data.iloc[:, 0]   # Labels (sign language digits)

X_test = test_data.iloc[:, 1:]  # Features (pixel values)
y_test = test_data.iloc[:, 0]   # Labels (sign language digits)

# Check for missing values
print("Missing values in train data:", X_train.isnull().sum().sum())
print("Missing values in test data:", X_test.isnull().sum().sum())

# Normalize pixel values to range [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape a sample image from the flattened format to a 28x28 format
sample_image = X_train[10].reshape(28, 28)  # Images are 28x28 pixels

# Plotting the image
plt.imshow(sample_image, cmap='gray')
plt.title(f"Label: {y_train[10]}")
plt.show()

# Initialize variables
k_values = list(range(1, 11))  # Range of K values to evaluate
accuracies = []

# Iterate over different K values
for k in k_values:
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on test data and compute accuracy
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plotting the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)  # You can experiment with different values of K	

knn.fit(X_train, y_train)

y_pred_test = knn.predict(X_test)

accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test, average='weighted')
recall_test = recall_score(y_test, y_pred_test, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')

print(f"Test Accuracy with Optimal K: {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1-Score: {f1_test:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print(classification_report(y_test, y_pred_test))