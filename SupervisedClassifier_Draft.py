# SVM classiifer

import rasterio
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from rasterio.enums import Resampling
from sklearn.utils import class_weight

# ----------------------------
# 1. Load GeoTIFF Data with NoData Handling
# ----------------------------
def load_geotiff(filepath):
    print(f"Loading raster from: {filepath}")
    with rasterio.open(filepath) as src:
        data = src.read().astype(np.float32)
        profile = src.profile
        nodata = src.nodata

        if nodata is not None:
            print(f"Detected NoData value: {nodata}")
            data[data == nodata] = np.nan
        else:
            fallback = -3.4028235e+38
            count = np.sum(data == fallback)
            if count > 0:
                print(f"Replacing fallback NoData value ({fallback}) in {count} cells")
                data[data == fallback] = np.nan

    print(f"Loaded raster with shape: {data.shape}")
    return data, profile

# ----------------------------
# 2. Resample Label Raster to Match Feature Raster
# ----------------------------
def resample_label_to_match_feature(label_path, feature_path):
    print(f"Resampling label raster: {label_path} to match feature raster: {feature_path}")
    with rasterio.open(feature_path) as feature_src:
        feature_height = feature_src.height
        feature_width = feature_src.width
        with rasterio.open(label_path) as label_src:
            label_resampled = label_src.read(
                out_shape=(1, feature_height, feature_width),
                resampling=Resampling.nearest
            )
            print(f"Resampled label shape: {label_resampled.shape}")
            return label_resampled[0]  # Return single-band array

# ----------------------------
# 3. Reshape Features
# ----------------------------
def reshape_features(image):
    bands, height, width = image.shape
    print(f"Reshaping features from (bands, height, width) = {image.shape}")
    return image.reshape(bands, height * width).T

# ----------------------------
# 4. Load and Prepare Data
# ----------------------------
feature_raster_path = "Imagery/RADARSATTest/RADARSAT_VVHHNormalizedRatio.tif"
label_raster_path = "TrainingData/TrainingData.tif"

feature_data, feature_profile = load_geotiff(feature_raster_path)
label_resampled = resample_label_to_match_feature(label_raster_path, feature_raster_path)

X = reshape_features(feature_data)
y = label_resampled.flatten()

print(f"Flattened feature matrix: {X.shape}")
print(f"Flattened labels array: {y.shape}")
print(f"Label unique values BEFORE filtering: {np.unique(y)}")

# ----------------------------
# 5. Filter Out Invalid Labels
# ----------------------------
valid_mask = (y >= 0)
X = X[valid_mask]
y = y[valid_mask]
print(f"After removing invalid labels: X shape = {X.shape}, y shape = {y.shape}")

# ----------------------------
# 6. Drop Rows Where Feature is NaN
# ----------------------------
nan_mask = ~np.isnan(X).any(axis=1)
X = X[nan_mask]
y = y[nan_mask]
print(f"After dropping NaN feature rows: X shape = {X.shape}, y shape = {y.shape}")

print("Checking feature value stats after filtering:")
print("Min:", np.nanmin(X[:, 0]), "Max:", np.nanmax(X[:, 0]), "Mean:", np.nanmean(X[:, 0]))

# ----------------------------
# 7. Class Distribution Check
# ----------------------------
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# ----------------------------
# 8. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("y_train label counts:", dict(zip(*np.unique(y_train, return_counts=True))))
print("y_test label counts:", dict(zip(*np.unique(y_test, return_counts=True))))

# ----------------------------
# 9. Train the SVM Classifier (with balanced classes)
# ----------------------------
print("Training SVM classifier...")
start_time = time.time()

clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
clf.fit(X_train, y_train)

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# ----------------------------
# 10. Evaluate Classifier
# ----------------------------
y_pred = clf.predict(X_test)

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

joblib.dump(clf, "sea_ice_classifier_svm.pkl")
print("Model saved to 'sea_ice_classifier_svm.pkl'.")

# ----------------------------
# 11. Apply Model to Full Raster (Batch Prediction)
# ----------------------------
print("Predicting full raster in batches...")
X_full = reshape_features(feature_data)

# Update mask to exclude pixels with NaN or value = 0
zero_mask = (X_full == 0).any(axis=1)
nan_mask = np.isnan(X_full).any(axis=1)
full_valid_mask = ~(nan_mask | zero_mask)

X_full_valid = X_full[full_valid_mask]
print(f"Full raster: {X_full.shape[0]} pixels total, {X_full_valid.shape[0]} valid for prediction")

# Set NoData as 255
y_full_pred = np.full(X_full.shape[0], 255, dtype=np.uint8)

batch_size = 50000
predictions = []
num_batches = int(np.ceil(X_full_valid.shape[0] / batch_size))

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, X_full_valid.shape[0])
    print(f"Predicting batch {i+1}/{num_batches}: indices {start_idx} to {end_idx}")
    batch_pred = clf.predict(X_full_valid[start_idx:end_idx])
    predictions.append(batch_pred)

predictions = np.concatenate(predictions)

# Insert predictions into the full prediction array
y_full_pred[full_valid_mask] = predictions

# Reshape to raster
height, width = feature_data.shape[1], feature_data.shape[2]
classified_image = y_full_pred.reshape((height, width))
print(f"Classified image reshaped to: {classified_image.shape}")

# ----------------------------
# 12. Save Classified Image as GeoTIFF
# ----------------------------
output_path = "Imagery/classified_image_svm2.tif"

# Delete existing file if it exists
import os
if os.path.exists(output_path):
    print(f"Deleting existing file: {output_path}")
    os.remove(output_path)

feature_profile.update({
    "dtype": rasterio.uint8,
    "count": 1,
    "nodata": 255
})

print(f"Saving classified raster to: {output_path}")
with rasterio.open(output_path, "w", **feature_profile) as dst:
    dst.write(classified_image.astype(rasterio.uint8), 1)

print("Classified image saved.")

# ----------------------------
# 13. Display Result
# ----------------------------
plt.figure(figsize=(8, 6))
plt.imshow(np.ma.masked_where(classified_image == 255, classified_image),
           cmap="coolwarm", vmin=0, vmax=1)
plt.title("Classified Image: 0 = Snow, 1 = Slush")
plt.colorbar(label="Class")
plt.tight_layout()
plt.show()
