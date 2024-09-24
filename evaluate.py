# Evaluate the trained model on ground-level images and compute localization error.
# Usage: python evaluate.py --use_gpu --checkpoint_path checkpoints/best_model_epoch_14.pth --gps_data_file datasets/trevor_multisense_forward_aux_image_rect_color_interpolated_gps.csv

import os
import argparse
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

# Function to extract timestamp from filename
def extract_timestamp(filename):
    basename = os.path.basename(filename)
    # Adjust this function based on your filename format
    if basename.startswith('frame_'):
        timestamp_str = basename.replace('frame_', '').replace('.jpg', '')
    elif basename.startswith('cropped_frame_'):
        timestamp_str = basename.replace('cropped_frame_', '').replace('.jpg', '')
    else:
        timestamp_str = None
    return timestamp_str

# Custom Dataset for ground-level images
class GroundDataset(Dataset):
    def __init__(self, ground_image_dir, transform=None):
        self.image_files = glob(os.path.join(ground_image_dir, '*.jpg'))
        self.transform = transform

        # Create a list of tuples (timestamp, image_path)
        self.images = []
        for img_path in self.image_files:
            timestamp = extract_timestamp(img_path)
            if timestamp:
                self.images.append((timestamp, img_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        timestamp, img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return timestamp, image

# Custom Dataset for aerial images
class AerialDataset(Dataset):
    def __init__(self, aerial_image_dir, transform=None):
        self.image_files = glob(os.path.join(aerial_image_dir, '*.jpg'))
        self.transform = transform

        # Create a list of tuples (timestamp, image_path)
        self.images = []
        for img_path in self.image_files:
            timestamp = extract_timestamp(img_path)
            if timestamp:
                self.images.append((timestamp, img_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        timestamp, img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return timestamp, image

# Embedding network (same as in train.py)
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        # Load pre-trained ResNet-50
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the last fully connected layer
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        # Add a new fully connected layer for projection
        self.fc = nn.Linear(2048, embedding_dim)
        # Initialize the weights
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # Normalize embeddings
        return x

# Cross-view model (same as in train.py)
class CrossViewModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super(CrossViewModel, self).__init__()
        self.ground_net = EmbeddingNet(embedding_dim)
        self.aerial_net = EmbeddingNet(embedding_dim)

    def forward(self, ground_img, aerial_img):
        if ground_img is not None:
            ground_embed = self.ground_net(ground_img)
        else:
            ground_embed = None
        if aerial_img is not None:
            aerial_embed = self.aerial_net(aerial_img)
        else:
            aerial_embed = None
        return ground_embed, aerial_embed

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Cross-View Matching Evaluation Script')
    parser.add_argument('--ground_image_dir', type=str, default='datasets/ground_level_images',
                        help='Directory containing ground-level images.')
    parser.add_argument('--aerial_image_dir', type=str, default='datasets/aerial_image_patches_16',
                        help='Directory containing aerial image patches.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.pth',
                        help='Path to the trained model checkpoint.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the embedding vector.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('--gps_data_file', type=str, default='gps_data.csv',
                        help='CSV file containing GPS data with timestamps.')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f'Using device: {device}')

    # Load GPS data
    import pandas as pd
    gps_df = pd.read_csv(args.gps_data_file)
    gps_df['timestamp'] = gps_df['timestamp'].astype(str)
    gps_data = gps_df.set_index('timestamp').to_dict('index')

    # Define transformations (same as in train.py, but without data augmentation)
    transform_ground = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_aerial = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the model
    model = CrossViewModel(embedding_dim=args.embedding_dim)
    # Load the state_dict
    state_dict = torch.load(args.checkpoint_path, map_location=device)

    # Remove 'module.' prefix if present
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove 'module.' prefix
        new_state_dict[name] = v

    # Load the adjusted state_dict
    model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()

    # Create datasets
    ground_dataset = GroundDataset(args.ground_image_dir, transform=transform_ground)
    aerial_dataset = AerialDataset(args.aerial_image_dir, transform=transform_aerial)

    # Create DataLoaders
    ground_loader = DataLoader(ground_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
    aerial_loader = DataLoader(aerial_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    # Step 1: Compute aerial embeddings
    print('Computing aerial embeddings...')
    aerial_embeddings = []
    aerial_timestamps = []
    aerial_coords = []

    with torch.no_grad():
        for timestamps, images in tqdm(aerial_loader):
            images = images.to(device, non_blocking=True)
            _, aerial_embed = model(None, images)
            aerial_embeddings.append(aerial_embed.cpu())
            aerial_timestamps.extend(timestamps)

    # Concatenate all embeddings
    aerial_embeddings = torch.cat(aerial_embeddings, dim=0).numpy()

    # Build KDTree for efficient nearest neighbor search
    print('Building KDTree for aerial embeddings...')
    kd_tree = KDTree(aerial_embeddings)


    # After computing aerial_embeddings and aerial_timestamps
    valid_aerial_embeddings = []
    valid_aerial_coords = []

    for i, timestamp in enumerate(aerial_timestamps):
        gps_info = gps_data.get(timestamp)
        if gps_info:
            coord = (gps_info['lat'], gps_info['lon'])
            valid_aerial_coords.append(coord)
            valid_aerial_embeddings.append(aerial_embeddings[i])
        else:
            # Skip this aerial image
            pass  # Optionally, keep track of skipped images

    # Convert lists to numpy arrays
    valid_aerial_embeddings = np.vstack(valid_aerial_embeddings)
    valid_aerial_coords = np.array(valid_aerial_coords)

    # Build KDTree with valid embeddings
    print('Building KDTree for aerial embeddings...')
    kd_tree = KDTree(valid_aerial_embeddings)



    # Get GPS coordinates for aerial images
    for timestamp in aerial_timestamps:
        gps_info = gps_data.get(timestamp)
        if gps_info:
            aerial_coords.append((gps_info['lat'], gps_info['lon']))
        else:
            # Handle missing GPS data
            aerial_coords.append((None, None))

    # Step 2: Evaluate on ground-level images
    print('Evaluating on ground-level images...')
    errors = []
    num_images = len(ground_dataset)

    with torch.no_grad():
        for timestamps, images in tqdm(ground_loader):
            images = images.to(device, non_blocking=True)
            ground_embed, _ = model(images, None)
            ground_embed = ground_embed.cpu().numpy()

            # For each ground embedding, find the nearest aerial embedding
            distances, indices = kd_tree.query(ground_embed, k=1)

            for i in range(len(timestamps)):
                ground_timestamp = timestamps[i]
                estimated_index = indices[i]
                estimated_coord = aerial_coords[estimated_index]

                # Get ground truth GPS coordinates
                gps_info = gps_data.get(ground_timestamp)
                if gps_info:
                    true_coord = (gps_info['lat'], gps_info['lon'])
                else:
                    continue  # Skip if no GPS data

                # Calculate localization error
                error = haversine_distance(true_coord, estimated_coord)
                errors.append(error)

    # Compute evaluation metrics
    errors = np.array(errors)
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    print(f'Mean Localization Error: {mean_error:.2f} meters')
    print(f'Median Localization Error: {median_error:.2f} meters')

    # Optionally, plot cumulative error distribution
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        sorted_errors = np.sort(errors)
        cdf = np.arange(len(sorted_errors)) / float(len(sorted_errors))
        plt.plot(sorted_errors, cdf)
        plt.xlabel('Localization Error (meters)')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Error Distribution')
        plt.grid(True)
        plt.savefig('/home/zhiyundeng/Cross-View-Matching/evaluation.png')  # Specify the path to save the image
        # plt.show()
    except ImportError:
        print('matplotlib not installed. Skipping plot.')

def haversine_distance(coord1, coord2):
    """
    Calculates the great-circle distance between two points on the Earth surface.
    Coordinates are given as (latitude, longitude) pairs in decimal degrees.
    """
    from math import radians, sin, cos, sqrt, atan2

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Handle missing coordinates
    if None in coord1 or None in coord2:
        return None

    # Earth radius in meters
    R = 6371000

    # Convert decimal degrees to radians
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    # Haversine formula
    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

if __name__ == '__main__':
    main()
