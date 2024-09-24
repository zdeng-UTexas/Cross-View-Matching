import os
import numpy as np
from osgeo import gdal
from geopy.distance import geodesic
import pandas as pd
import matplotlib.pyplot as plt

def get_project_path():
    # Get the current working directory
    current_dir = os.getcwd()

    # Get the project directory by going up one level
    project_dir = os.path.dirname(current_dir)

    return project_dir

def load_npy_file(npy_file_path):
    # Load the .npy file
    data = np.load(npy_file_path)
    return data

def reshape_data(data):
    # Reshape the data to 2D if necessary for CSV (flattening the 3D array to 2D)
    print(f"Data shape: {data.shape}")
    reshaped_data = data.reshape(-1, data.shape[2])
    return reshaped_data

def convert_to_dataframe(data):
    # Convert the numpy array to a Pandas DataFrame
    df = pd.DataFrame(data, columns=['R', 'G', 'B', 'Latitude', 'Longitude'])
    return df

def save_to_csv(df, csv_file_path):
    # Save the DataFrame to CSV
    df.to_csv(csv_file_path, index=False)

def normalize_rgb_data(rgb_data):
    # Normalize the RGB data if values exceed the range [0, 255]
    if rgb_data.max() > 255:
        rgb_data = rgb_data / rgb_data.max()  # Normalizing to the range [0, 1]
    elif rgb_data.max() > 1:
        rgb_data = rgb_data / 255  # If values are > 1 but <= 255, normalize to [0, 1]
    return rgb_data

def plot_visualization(rgb_data, latitude_data, longitude_data, output_image_path):
    # Create a figure for displaying the RGB image and heatmaps
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the RGB image
    ax[0].imshow(rgb_data)
    ax[0].set_title('RGB Image')

    # Plot the latitude heatmap
    lat_img = ax[1].imshow(latitude_data, cmap='coolwarm')
    ax[1].set_title('Latitude Variation')
    fig.colorbar(lat_img, ax=ax[1])

    # Plot the longitude heatmap
    lon_img = ax[2].imshow(longitude_data, cmap='coolwarm')
    ax[2].set_title('Longitude Variation')
    fig.colorbar(lon_img, ax=ax[2])

    # Save the figure
    plt.savefig(output_image_path)
    print(f"Visualization saved to {output_image_path}")

def compute_meters_per_pixel(latitude_data, longitude_data):
    # Calculate the distance between two consecutive pixels in the horizontal direction (same row)
    lat1, lon1 = latitude_data[0, 0], longitude_data[0, 0]  # First pixel
    lat2, lon2 = latitude_data[0, 1], longitude_data[0, 1]  # Second pixel (right)
    
    # Calculate distance between two consecutive pixels in the vertical direction (same column)
    lat3, lon3 = latitude_data[0, 0], longitude_data[0, 0]  # First pixel
    lat4, lon4 = latitude_data[1, 0], longitude_data[1, 0]  # Pixel below

    # Compute distances in meters
    horizontal_distance = geodesic((lat1, lon1), (lat2, lon2)).meters
    vertical_distance = geodesic((lat3, lon3), (lat4, lon4)).meters

    return horizontal_distance, vertical_distance

def compute_meters_per_pixel_using_corners(latitude_data, longitude_data):
    # Get the coordinates of the first pixel (top-left corner)
    lat1, lon1 = latitude_data[0, 0], longitude_data[0, 0]

    # Get the coordinates of the last pixel in the horizontal direction (top-right corner)
    lat2, lon2 = latitude_data[0, -1], longitude_data[0, -1]

    # Get the coordinates of the last pixel in the vertical direction (bottom-left corner)
    lat3, lon3 = latitude_data[-1, 0], longitude_data[-1, 0]

    # Calculate the geodesic distance (in meters) between the first and last pixel horizontally
    horizontal_distance = geodesic((lat1, lon1), (lat2, lon2)).meters

    # Calculate the geodesic distance (in meters) between the first and last pixel vertically
    vertical_distance = geodesic((lat1, lon1), (lat3, lon3)).meters

    # Calculate the pixel resolution by dividing the total distance by the number of pixels
    horizontal_resolution = horizontal_distance / (latitude_data.shape[1] - 1)  # Number of horizontal pixels
    vertical_resolution = vertical_distance / (latitude_data.shape[0] - 1)  # Number of vertical pixels

    return horizontal_resolution, vertical_resolution

def main():
    project_dir = get_project_path()
    npy_file_path = os.path.join(project_dir, 'datasets/geodata/gq-map.npy')
    tiff_file_path = os.path.join(project_dir, 'datasets/gq-map.tiff')
    csv_file_path = os.path.join(project_dir, 'datasets/geodata/gq-map.csv')
    output_image_path = os.path.join(project_dir, 'datasets/geodata/lat_lon_visualization.png')

    # Load the .npy file
    data = load_npy_file(npy_file_path)

    # Reshape the data
    reshaped_data = reshape_data(data)

    # Convert to DataFrame
    df = convert_to_dataframe(reshaped_data)

    # Save DataFrame to CSV
    # save_to_csv(df, csv_file_path)
    # print(f"Data saved to {csv_file_path}")

    # Extract RGB channels and latitude/longitude
    rgb_data = data[:, :, :3]
    latitude_data = data[:, :, 3]
    longitude_data = data[:, :, 4]

    # Normalize the RGB data
    rgb_data = normalize_rgb_data(rgb_data)

    # Plot visualization
    plot_visualization(rgb_data, latitude_data, longitude_data, output_image_path)

    # Compute meters per pixel based on the latitude and longitude data in the file
    # horizontal_meters_per_pixel, vertical_meters_per_pixel = compute_meters_per_pixel(latitude_data, longitude_data)
    # print(f"Estimated meters per pixel - Horizontal: {horizontal_meters_per_pixel:.2f} meters, Vertical: {vertical_meters_per_pixel:.2f} meters")

    # Compute the real-world size of a pixel using the first and last pixels
    horizontal_meters_per_pixel, vertical_meters_per_pixel = compute_meters_per_pixel_using_corners(latitude_data, longitude_data)
    print(f"Estimated meters per pixel - Horizontal: {horizontal_meters_per_pixel:.2f} meters, Vertical: {vertical_meters_per_pixel:.2f} meters")


if __name__ == "__main__":
    main()
