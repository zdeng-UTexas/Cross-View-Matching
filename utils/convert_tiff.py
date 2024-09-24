import os
import numpy as np
import pandas as pd
from PIL import Image
import argparse

# Argument parser for patch size
parser = argparse.ArgumentParser(description="Crop patches from aerial image using GPS coordinates.")
parser.add_argument('--patch_size', type=int, required=True, help="Size of the patch to crop (e.g., 32 for 32x32 patch)")
args = parser.parse_args()

# Get the current working directory and locate relevant paths dynamically
project_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(project_dir, '..', 'datasets')
geodata_path = os.path.join(datasets_dir, 'geodata', 'gq-map.npy')
aerial_image_path = os.path.join(datasets_dir, 'gq-map.tiff')
gps_csv_path = os.path.join(datasets_dir, 'trevor_multisense_forward_aux_image_rect_color_interpolated_gps.csv')
ground_images_dir = os.path.join(datasets_dir, 'ground_level_images')

# Load aerial map and geodata
aerial_image = Image.open(aerial_image_path)
geodata = np.load(geodata_path)

# Convert geodata to DataFrame for easier manipulation
df = pd.DataFrame(geodata.reshape(-1, 5), columns=['R', 'G', 'B', 'Latitude', 'Longitude'])

# Load GPS CSV file
gps_data = pd.read_csv(gps_csv_path)

# Function to find the closest GPS coordinate from the CSV file
def find_gps_from_timestamp(image_filename):
    timestamp_str = image_filename.split('_')[1].split('.')[0]
    timestamp = float(timestamp_str)
    closest_row = gps_data.iloc[(gps_data['timestamp'] - timestamp).abs().idxmin()]
    return closest_row['lat'], closest_row['lon']

# Function to find the closest pixel for a given GPS coordinate
def find_closest_pixel(lat, lon):
    distances = np.sqrt((df['Latitude'] - lat)**2 + (df['Longitude'] - lon)**2)
    idx_closest = distances.idxmin()
    row, col = divmod(idx_closest, geodata.shape[1])  # geodata.shape[1] is the number of columns in the original aerial image
    return row, col

# Function to crop a patch from the aerial image centered on a given GPS coordinate
def crop_patch_from_aerial(lat, lon, size):
    row, col = find_closest_pixel(lat, lon)
    left = max(0, col - size // 2)
    upper = max(0, row - size // 2)
    right = min(aerial_image.width, col + size // 2)
    lower = min(aerial_image.height, row + size // 2)
    
    # Crop and return the patch
    cropped_image = aerial_image.crop((left, upper, right, lower))
    return cropped_image

# Create directory to save cropped images
def create_output_directory(size):
    output_dir = os.path.join(datasets_dir, f'positive_samples_{size}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Process ground-level images and crop patches from the aerial image
def process_ground_images(patch_size):
    for image_file in os.listdir(ground_images_dir):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(ground_images_dir, image_file)
            
            # Find the corresponding GPS coordinates
            lat, lon = find_gps_from_timestamp(image_file)
            
            # Crop the image patch from the aerial image
            cropped_image = crop_patch_from_aerial(lat, lon, patch_size)
            
            # Convert to RGB if necessary
            if cropped_image.mode == 'RGBA':
                cropped_image = cropped_image.convert('RGB')
            
            # Save the cropped image in the positive_samples_size directory
            output_dir = create_output_directory(patch_size)
            output_image_path = os.path.join(output_dir, f'cropped_{image_file}')
            cropped_image.save(output_image_path, format='JPEG')

# Example usage with patch_size as an argument
process_ground_images(args.patch_size)