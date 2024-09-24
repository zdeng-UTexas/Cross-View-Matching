# The script crops multiple image patches from an aerial image centered around a GPS coordinate derived from a given timestamp, using specified patch size, stepsize, and scale.
# Usage: python utils/crop_aerial_image_candidates.py --timestamp 1723659171.84382 --aerial_image_path datasets/gq-map.tiff --patch_size 16 --stepsize 1 --scale 50

import os
import numpy as np
import pandas as pd
from PIL import Image
import argparse

# Function to find the GPS coordinate from the timestamp
def find_gps_from_timestamp(timestamp, gps_data):
    """
    Finds the GPS coordinates (latitude, longitude) corresponding to the given timestamp.

    Args:
        timestamp (float): The timestamp of the ground-level image.
        gps_data (pd.DataFrame): DataFrame containing timestamp and GPS information.

    Returns:
        tuple: (latitude, longitude) of the closest timestamp.
    """
    closest_row = gps_data.iloc[(gps_data['timestamp'] - timestamp).abs().idxmin()]
    return closest_row['lat'], closest_row['lon']

# Function to find the closest pixel for a given GPS coordinate
def find_closest_pixel(lat, lon, geodata):
    """
    Finds the closest pixel coordinates in the image for given GPS coordinates.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        geodata (numpy.ndarray): Georeferenced data with shape (rows, cols, 5), where 5 channels are R, G, B, Latitude, Longitude.

    Returns:
        row (int): Row index in the image.
        col (int): Column index in the image.
    """
    distances = np.sqrt((geodata[:, :, 3] - lat)**2 + (geodata[:, :, 4] - lon)**2)
    idx_closest = distances.argmin()
    row, col = np.unravel_index(idx_closest, geodata.shape[:2])
    return row, col

# Function to crop a patch from the aerial image
def crop_patch_from_aerial(aerial_image, row, col, size):
    """
    Crops a patch from the aerial image centered on the given pixel coordinates.

    Args:
        aerial_image (PIL.Image): The aerial image.
        row (int): Row index of the patch center.
        col (int): Column index of the patch center.
        size (int): Size of the patch in pixels.

    Returns:
        PIL.Image: The cropped patch image.
    """
    left = max(0, col - size // 2)
    upper = max(0, row - size // 2)
    right = min(aerial_image.width, col + size // 2)
    lower = min(aerial_image.height, row + size // 2)
    
    cropped_image = aerial_image.crop((left, upper, right, lower))
    return cropped_image

# Function to crop candidate patches
def crop_candidate_patches(aerial_image, geodata, center_coord, patch_size, stepsize, scale):
    """
    Crops candidate patches from an aerial image around a central GPS coordinate.

    Args:
        aerial_image (PIL.Image): The aerial image.
        geodata (numpy.ndarray): Georeferenced data with shape (rows, cols, 5), where 5 channels are R, G, B, Latitude, Longitude.
        center_coord (tuple): (latitude, longitude) of the center point.
        patch_size (int): Size of each patch in pixels.
        stepsize (int): Step size between patches.
        scale (int): Number of patches in each direction around the center.

    Returns:
        candidate_patches (list of PIL.Image): List of cropped patches.
        candidate_coords (list of tuple): List of (latitude, longitude) coordinates of each patch.
    """
    # Initialize lists for patches and coordinates
    candidate_patches = []
    candidate_coords = []

    # Find the center pixel in the aerial image
    center_row, center_col = find_closest_pixel(center_coord[0], center_coord[1], geodata)

    # Calculate half scale and steps
    half_scale = scale // 2
    half_size = patch_size // 2

    # Loop over grid defined by scale
    for i in range(-half_scale, half_scale + 1):
        for j in range(-half_scale, half_scale + 1):
            # Calculate new pixel coordinates for the patch center
            new_row = center_row + i * stepsize
            new_col = center_col + j * stepsize
            
            # Crop the patch
            cropped_patch = crop_patch_from_aerial(aerial_image, new_row, new_col, patch_size)
            
            # Get the GPS coordinates of the patch center
            patch_lat = geodata[new_row, new_col, 3]
            patch_lon = geodata[new_row, new_col, 4]
            
            # Store the patch and coordinates
            candidate_patches.append(cropped_patch)
            candidate_coords.append((patch_lat, patch_lon))

    return candidate_patches, candidate_coords

# Only execute this part when running the script directly
if __name__ == '__main__':
    # Argument parser for input parameters
    parser = argparse.ArgumentParser(description="Crop multiple image patches from aerial image using GPS coordinates and stepsize.")
    parser.add_argument('--timestamp', type=float, required=True, help="Timestamp of the ground-level image.")
    parser.add_argument('--aerial_image_path', type=str, required=True, help="Path to the aerial image in TIFF format.")
    parser.add_argument('--patch_size', type=int, required=True, help="Size of the patch to crop (e.g., 16 for 16x16 patch).")
    parser.add_argument('--stepsize', type=int, required=True, help="Stepsize for moving the window (e.g., 2 pixels).")
    parser.add_argument('--scale', type=int, required=True, help="Scale factor for grid (e.g., 5 means 5x5 patches).")
    args = parser.parse_args()
    
    # Get the current working directory and locate relevant paths dynamically
    project_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(project_dir, '..', 'datasets')
    gps_csv_path = os.path.join(datasets_dir, 'trevor_multisense_forward_aux_image_rect_color_interpolated_gps.csv')
    ground_images_dir = os.path.join(datasets_dir, 'ground_level_images')

    # Load aerial image and geodata
    aerial_image = Image.open(args.aerial_image_path).convert('RGB')
    geodata_path = os.path.join(datasets_dir, 'geodata', 'gq-map.npy')
    geodata = np.load(geodata_path)

    # Load GPS CSV file
    gps_data = pd.read_csv(gps_csv_path)
    
    # Find GPS coordinate from timestamp
    lat, lon = find_gps_from_timestamp(args.timestamp, gps_data)
    center_coord = (lat, lon)
    
    # Call the processing function
    candidate_patches, candidate_coords = crop_candidate_patches(
        aerial_image=aerial_image,
        geodata=geodata,
        center_coord=center_coord,
        patch_size=args.patch_size,
        stepsize=args.stepsize,
        scale=args.scale
    )
    
    # Save or process candidate patches as needed
    output_dir = 'output_patches'
    os.makedirs(output_dir, exist_ok=True)
    for idx, patch in enumerate(candidate_patches):
        patch.save(f'{output_dir}/patch_{idx}.jpg')
    print(f"Saved {len(candidate_patches)} patches to {output_dir}")
