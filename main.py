import numpy as np
import pandas as pd
import cv2
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def extract_image_features(file_path):
    """
    Reads an image from file_path, resizes it to 128x128, extracts the green channel,
    applies CLAHE, and computes multiple features:
      - GLCM features: contrast, energy, homogeneity, correlation
      - Histogram features: normalized histogram with 16 bins
      - Statistical features: mean and standard deviation
    Returns a combined feature vector.
    """
    # Read image in color
    image = cv2.imread(file_path)
    if image is None:
        return None

    # Resize to 128x128
    image_resized = cv2.resize(image, (128, 128))

    # Extract green channel (OpenCV uses BGR format)
    green_channel = image_resized[:, :, 1]

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)

    # Compute GLCM features
    glcm = graycomatrix(enhanced, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Compute normalized histogram features (16 bins)
    hist = cv2.calcHist([enhanced], [0], None, [16], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Compute statistical features: mean and std
    mean_intensity = np.mean(enhanced)
    std_intensity = np.std(enhanced)

    # Combine all features into a single vector
    features = [contrast, energy, homogeneity, correlation] + hist.tolist() + [mean_intensity, std_intensity]
    return features


# --------------------- Data Loading & Feature Extraction ---------------------
# Load the CSV file
data = pd.read_csv("messidor_data.csv")  # Ensure this file exists
image_folder = "preprocess"  # Folder containing images

features_list = []
labels_list = []

for idx, row in data.iterrows():
    # Get the image filename from CSV; if extension is missing, try adding one
    filename = row["id_code"]
    if not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg')):
        # If images are stored as PNG, you can change to '.png'
        filename += ".png"
    file_path = os.path.join(image_folder, filename)

    feats = extract_image_features(file_path)
    if feats is not None:
        features_list.append(feats)
        labels_list.append(row["diagnosis"])  # Adjust if your label column name differs

features_array = np.array(features_list)
labels_array = np.array(labels_list)

# --------------------- Data Augmentation / Balancing (SMOTE) ---------------------
# Use SMOTE to oversample minority classes if needed.
smote = SMOTE(random_state=42)
features_res, labels_res = smote.fit_resample(features_array, labels_array)

# --------------------- Train-Test Split ---------------------
X_train, X_test, y_train, y_test = train_test_split(features_res, labels_res, test_size=0.2, random_state=42)


#
# Print the confusion matrix using Matplotlib

# --------------------- Hyperparameter Tuning for Random Forest ---------------------
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 12, 15],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

rf_predictions = best_rf.predict(X_test)

# --------------------- Naïve Bayes Classifier ---------------------
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predictions = nb.predict(X_test)

# --------------------- Evaluation ---------------------
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_sensitivity = recall_score(y_test, rf_predictions, average='macro')
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_sensitivity = recall_score(y_test, nb_predictions, average='macro')

def calculate_specificity(cm):
    TN = cm[0, 0]
    FP = cm[0, 1]
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    return specificity

# Calculate specificity for each classifier
rf_cm = confusion_matrix(y_test, rf_predictions)
nb_cm = confusion_matrix(y_test, nb_predictions)
rf_specificity = calculate_specificity(rf_cm)
nb_specificity = calculate_specificity(nb_cm)

print("Optimized Random Forest Results:")
print(f"Accuracy: {rf_accuracy * 100:.2f}%")
print(f"Sensitivity: {rf_sensitivity * 100:.2f}%")
print(f"Random Forest Specificity: {rf_specificity * 100:.2f}%")

print("\nNaïve Bayes Results:")
print(f"Accuracy: {nb_accuracy * 100:.2f}%")
print(f"Sensitivity: {nb_sensitivity * 100:.2f}%")
print(f"Random Forest Specificity: {rf_specificity * 100:.2f}%")
print(f"Naïve Bayes Specificity: {nb_specificity * 100:.2f}%")
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_predictions))
print("\nNaïve Bayes Classification Report:\n", classification_report(y_test, nb_predictions))

# --------------------- Histogram (Feature Distribution) ---------------------
plt.figure(figsize=(10, 5))
plt.hist(features_res[:, 0], bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Feature Values')
plt.ylabel('Frequency')
plt.title('Histogram of First Feature')
plt.show()

# --------------------- Bar Chart (Model Accuracies) ---------------------
models = ['Random Forest', 'Naïve Bayes']
accuracies = [rf_accuracy * 100, nb_accuracy * 100]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies, palette='viridis')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracies')
plt.ylim(0, 100)
plt.show()

# --------------------- Line Plot (Feature Trends) ---------------------
plt.figure(figsize=(10, 5))
for i in range(5):  # Plot first 5 features
    plt.plot(features_res[:100, i], label=f'Feature {i+1}')

plt.xlabel('Sample Index')
plt.ylabel('Feature Value')
plt.title('Feature Trends Over Samples')
plt.legend()
plt.show()

# --------------------- Scatter Plot (Feature Relationship) ---------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=features_res[:, 0], y=features_res[:, 1], hue=labels_res, palette='coolwarm', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Two Features')
plt.legend(title='Class')
plt.show()





# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

# Plot for Random Forest
plot_confusion_matrix(rf_cm, "Confusion Matrix - Random Forest")

# Plot for Naïve Bayes
plot_confusion_matrix(nb_cm, "Confusion Matrix - Naïve Bayes")
