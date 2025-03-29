import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import os

# Load raster image
def load_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read(1)  # Reads first band (assuming VV/HH ratio)
        profile = src.profile
        nodata_value = src.nodata
    return image, profile, nodata_value

# Save classified raster image
def save_classification(classified_image, profile, output_path):
    profile.update(dtype=rasterio.int16, count=1, nodata=-3)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(classified_image, 1)

# Compute local GLCM texture features robustly
def compute_local_glcm_features(image, window_size=3):
    image_norm = (image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image))
    image_uint8 = img_as_ubyte(np.nan_to_num(image_norm))

    pad_size = window_size // 2
    padded_image = np.pad(image_uint8, pad_size, mode='edge')

    contrast = np.zeros(image.shape)
    correlation = np.zeros(image.shape)
    energy = np.zeros(image.shape)
    homogeneity = np.zeros(image.shape)

    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            window = padded_image[i:i+window_size, j:j+window_size]
            glcm = graycomatrix(window, distances=[1], angles=[0], symmetric=True, normed=True)
            contrast[i, j] = graycoprops(glcm, 'contrast')[0, 0]
            correlation[i, j] = graycoprops(glcm, 'correlation')[0, 0]
            energy[i, j] = graycoprops(glcm, 'energy')[0, 0]
            homogeneity[i, j] = graycoprops(glcm, 'homogeneity')[0, 0]

    features = np.stack([contrast, correlation, energy, homogeneity], axis=-1)
    return features

# Classify image with improved masking
def classify_image(image_data, nodata_value):
    valid_mask = (~np.isnan(image_data)) & (image_data != nodata_value) & (image_data != 0)

    texture_features = compute_local_glcm_features(image_data, window_size=3)

    feature_stack = np.dstack([image_data, texture_features])

    X = feature_stack[valid_mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    classified_image = np.full(image_data.shape, fill_value=-3, dtype=np.int16)
    classified_image[valid_mask] = labels

    return classified_image

# Process and visualize images with discrete legend
def process_images(input_dir, output_dir):
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]

    class_labels = {
        0: 'Class 0',
        1: 'Class 1',
        2: 'Class 2',
        3: 'Class 3',
        -3: 'No Data/Land'
    }

    cmap = plt.cm.get_cmap('tab10', 5)  # discrete colormap

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image, profile, nodata_value = load_image(image_path)

        print(f"Processing {image_file}...")

        classified_image = classify_image(image, nodata_value)

        plt.figure(figsize=(10, 8))
        im = plt.imshow(classified_image, cmap=cmap, vmin=-3, vmax=3)

        cbar = plt.colorbar(im, ticks=[-3, 0, 1, 2, 3])
        cbar.ax.set_yticklabels([
            class_labels[-3],
            class_labels[0],
            class_labels[1],
            class_labels[2],
            class_labels[3]
        ])
        plt.title(f'Classification of {image_file}')
        plt.xlabel('Pixel Column')
        plt.ylabel('Pixel Row')
        plt.show()

        output_path = os.path.join(output_dir, f'classified_{image_file}')
        save_classification(classified_image, profile, output_path)

        print(f"Completed {image_file}. Saved to {output_path}\n")

# Main execution
def main():
    input_dir = 'Imagery/RADARSATTest'
    output_dir = 'Imagery/Output'
    os.makedirs(output_dir, exist_ok=True)
    process_images(input_dir, output_dir)

if __name__ == '__main__':
    main()
