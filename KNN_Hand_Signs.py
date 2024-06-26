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

# Statistical summary
print("Train Data Summary:")
print(train_data.describe())

# Standardize pixel values using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape a sample image from the flattened format to a 28x28 format
sample_image = X_train[21].reshape(28, 28)  # Images are 28x28 pixels

# Plotting the image
plt.imshow(sample_image, cmap='gray')
plt.title(f"Label: {y_train[21]}")
plt.show()

# Fixed K value for all metrics
k = 3

# List of distance metrics to evaluate
metrics = ['euclidean', 'manhattan', 'chebyshev']

results = {}

for metric in metrics:
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train, y_train)

    # Predict on test data
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[metric] = accuracy

    # Print results
    print(f"\nMetric: {metric}")
    print(f"Accuracy: {accuracy:.4f}")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'Confusion Matrix ({metric})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Print classification report
    print(f"Classification Report (Metric: {metric}):\n")
    print(classification_report(y_test, y_pred))
    
    # Function to predict a single sample
def predict_single_sample(model, sample):
    sample = sample.reshape(1, -1)  # Reshape if necessary
    prediction = model.predict(sample)
    return prediction[0]

# Example of testing three samples
sample_indices = [0, 1, 2]
for sample_index in sample_indices:
    sample = X_test[sample_index]
    true_label = y_test[sample_index]

    # Reshape and plot the sample image
    sample_image = sample.reshape(28, 28)
    plt.imshow(sample_image, cmap='gray')
    plt.title(f"True Label: {true_label}")
    plt.show()

    print(f"\nTrue label for sample {sample_index}: {true_label}")

    for metric in metrics:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train, y_train)
        prediction = predict_single_sample(knn, sample)
        print(f"Prediction for sample {sample_index} with {metric} metric: {prediction}")
