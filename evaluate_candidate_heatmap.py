import os
import argparse
from glob import glob
from PIL import Image
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm


# Import your custom cropping function
from utils.crop_aerial_image_candidates import crop_candidate_patches

# Haversine distance function
def haversine_distance(coord1, coord2):
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

# Custom Dataset for ground-level images
class GroundDataset(Dataset):
    def __init__(self, ground_image_dir, transform=None):
        self.image_files = glob(os.path.join(ground_image_dir, '*.jpg'))
        self.transform = transform

        # Create a dictionary mapping timestamps to image paths
        self.images = {}
        for img_path in self.image_files:
            timestamp = extract_timestamp(img_path)
            if timestamp:
                self.images[timestamp] = img_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        timestamps = list(self.images.keys())
        timestamp = timestamps[idx]
        img_path = self.images[timestamp]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return timestamp, image

# Embedding network
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        # Use the recommended way to load pre-trained weights
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
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

# Cross-view model
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

def extract_timestamp(filename):
    basename = os.path.basename(filename)
    # Remove the prefixes 'frame_' or 'cropped_frame_'
    if basename.startswith('frame_'):
        timestamp_str = basename.replace('frame_', '').replace('.jpg', '')
    elif basename.startswith('cropped_frame_'):
        timestamp_str = basename.replace('cropped_frame_', '').replace('.jpg', '')
    else:
        timestamp_str = None
    return timestamp_str

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Cross-View Matching Evaluation with Localized Candidate Patches')
    parser.add_argument('--ground_image_dir', type=str, default='datasets/ground_level_images',
                        help='Directory containing ground-level images.')
    parser.add_argument('--aerial_image_path', type=str, required=True,
                        help='Path to the high-resolution aerial image.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_model.pth',
                        help='Path to the trained model checkpoint.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the embedding vector.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('--gps_data_file', type=str, default='gps_data.csv',
                        help='CSV file containing GPS data with timestamps.')
    # New parameters
    parser.add_argument('--patch_size', type=int, default=16, help='Size of aerial patches.')
    parser.add_argument('--stepsize', type=int, default=1, help='Stepsize for candidate patches.')
    parser.add_argument('--scale', type=int, default=50, help='Scale (in meters) around the central GPS coordinate.')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f'Using device: {device}')

    # Load GPS data
    import pandas as pd
    gps_df = pd.read_csv(args.gps_data_file)
    gps_df['timestamp'] = gps_df['timestamp'].astype(str)
    gps_data = gps_df.set_index('timestamp').to_dict('index')

    # Define transformations (without data augmentation)
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
    model = model.to(device)
    model.eval()

    # Adjust state_dict keys and load
    from collections import OrderedDict
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove 'module.' prefix
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    # Create dataset for ground-level images
    ground_dataset = GroundDataset(args.ground_image_dir, transform=transform_ground)
    ground_loader = DataLoader(ground_dataset, batch_size=1, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    # Prepare aerial image
    project_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(project_dir, 'datasets')
    aerial_image = Image.open(args.aerial_image_path).convert('RGB')
    geodata_path = os.path.join(datasets_dir, 'geodata', 'gq-map.npy')
    geodata = np.load(geodata_path)
    
    errors = []
    num_images = len(ground_dataset)

    # # Create directory to save heatmaps
    # heatmaps_dir = os.path.join(project_dir, 'heatmaps')
    # os.makedirs(heatmaps_dir, exist_ok=True)

    # Create directory to save visualizations
    visualizations_dir = os.path.join(project_dir, 'visualizations', f'{args.patch_size}_{args.stepsize}_{args.scale}')
    os.makedirs(visualizations_dir, exist_ok=True)

    # For each ground-level image
    with torch.no_grad():
        for timestamps, images in tqdm(ground_loader):
            timestamp = timestamps[0]
            ground_img = images.to(device, non_blocking=True)
            # Get ground truth GPS coordinates
            gps_info = gps_data.get(timestamp)
            if gps_info:
                true_coord = (gps_info['lat'], gps_info['lon'])
            else:
                continue  # Skip if no GPS data

            # Generate candidate patches
            candidate_patches, candidate_coords = crop_candidate_patches(
                aerial_image=aerial_image,
                geodata=geodata,
                center_coord=true_coord,
                patch_size=args.patch_size,
                stepsize=args.stepsize,
                scale=args.scale
            )

            if len(candidate_patches) == 0:
                continue  # Skip if no candidate patches generated

            # Compute embeddings for candidate patches
            candidate_embeddings = []
            for patch in candidate_patches:
                if transform_aerial:
                    patch_tensor = transform_aerial(patch)
                else:
                    patch_tensor = transforms.ToTensor()(patch)
                patch_tensor = patch_tensor.unsqueeze(0).to(device)
                _, aerial_embed = model(None, patch_tensor)
                candidate_embeddings.append(aerial_embed.cpu())
            candidate_embeddings = torch.cat(candidate_embeddings, dim=0)  # Shape: [num_candidates, embedding_dim]

            # Compute embedding for ground image
            ground_embed, _ = model(ground_img, None)
            ground_embed = ground_embed.cpu()  # Shape: [1, embedding_dim]

            # Compute similarity scores
            similarity_scores = torch.nn.functional.cosine_similarity(
                ground_embed, candidate_embeddings, dim=1
            )  # Shape: [num_candidates]

            similarity_scores = similarity_scores.numpy()  # Convert to NumPy array

            # Determine grid size
            num_patches = len(candidate_patches)
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size != num_patches:
                print(f"Number of candidate patches ({num_patches}) is not a perfect square.")
                continue  # Skip or handle accordingly

            # Reshape similarity scores into a grid
            similarity_grid = similarity_scores.reshape((grid_size, grid_size))

            # Arrange candidate patches into a grid
            patch_grid = []
            idx = 0
            for _ in range(grid_size):
                row_patches = []
                for _ in range(grid_size):
                    row_patches.append(candidate_patches[idx])
                    idx += 1
                patch_grid.append(row_patches)

            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            # Define the figure and GridSpec with equal height ratios
            fig = plt.figure(figsize=(20, 6))  # Adjust the figure size as needed
            # Set both width and height ratios to be equal for the subplots
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], height_ratios=[1])

            # Ground-level image subplot
            ax_ground = fig.add_subplot(gs[0, 0])

            # Load the original ground-level image
            img_path = ground_dataset.images[timestamp]
            original_img = Image.open(img_path).convert('RGB')
            ground_img_np = np.array(original_img) / 255.0  # Normalize pixel values to [0, 1]

            ax_ground.imshow(ground_img_np)
            ax_ground.set_title('Ground-Level Image')
            ax_ground.axis('off')

            # Aerial patches grid
            # Create a GridSpec within the main GridSpec for the patches grid
            gs_patches = gridspec.GridSpecFromSubplotSpec(
                grid_size, grid_size, subplot_spec=gs[0, 1], wspace=0.0, hspace=0.0)

            for i in range(grid_size):
                for j in range(grid_size):
                    ax = fig.add_subplot(gs_patches[i, j])
                    patch_np = np.array(patch_grid[i][j]) / 255.0  # Normalize pixel values
                    ax.imshow(patch_np)
                    ax.axis('off')

            # Set title for the aerial patches grid
            ax_patches_title = fig.add_subplot(gs[0, 1])
            ax_patches_title.axis('off')
            ax_patches_title.set_title('Aerial Patches Grid')

            # Display heatmap with 'viridis' colormap
            ax_heatmap = fig.add_subplot(gs[0, 2])
            im = ax_heatmap.imshow(similarity_grid, cmap='viridis', interpolation='nearest')  # Use viridis colormap
            ax_heatmap.set_title('Similarity Heatmap', color='black')  # Optional: Set title color
            ax_heatmap.set_xlabel('X-axis', color='black')  # Optional: Set x-axis label color
            ax_heatmap.set_ylabel('Y-axis', color='black')  # Optional: Set y-axis label color
            ax_heatmap.invert_yaxis()

            # Add colorbar for the heatmap
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax_heatmap)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            plt.tight_layout()
            plt.savefig(os.path.join(visualizations_dir, f'visualization_{timestamp}.png'))
            plt.close()


            # Find the position with the highest similarity score
            max_index = np.argmax(similarity_scores)
            estimated_coord = candidate_coords[max_index]

            # Calculate localization error
            error = haversine_distance(true_coord, estimated_coord)
            if error is not None:
                errors.append(error)
            else:
                # Optionally, keep track of failed calculations
                pass

    # Compute evaluation metrics
    errors = [e for e in errors if e is not None]

    if len(errors) == 0:
        print('No valid errors to compute statistics.')
    else:
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
            plt.savefig(f'evaluation_candidate_{args.patch_size}_{args.stepsize}_{args.scale}.png')
            # plt.show()
        except ImportError:
            print('matplotlib not installed. Skipping plot.')

if __name__ == '__main__':
    main()
