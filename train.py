import os
import random
import argparse
from glob import glob
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

def extract_timestamp(filename):
    """
    Extracts the timestamp from the filename.
    Adjust this function based on your filename format.
    """
    basename = os.path.basename(filename)
    # Remove the prefixes 'frame_' or 'cropped_frame_'
    if basename.startswith('frame_'):
        timestamp_str = basename.replace('frame_', '').replace('.jpg', '')
    elif basename.startswith('cropped_frame_'):
        timestamp_str = basename.replace('cropped_frame_', '').replace('.jpg', '')
    else:
        # Handle unexpected filename formats
        timestamp_str = None
    return timestamp_str

def create_triplets(ground_image_dir, aerial_image_dir):
    """
    Creates a list of triplets (anchor, positive, negative) based on timestamps.
    """
    # Get lists of image files
    ground_image_files = glob(os.path.join(ground_image_dir, '*.jpg'))
    aerial_image_files = glob(os.path.join(aerial_image_dir, '*.jpg'))

    # Create dictionaries mapping timestamps to file paths
    ground_images = {extract_timestamp(f): f for f in ground_image_files}
    aerial_images = {extract_timestamp(f): f for f in aerial_image_files}

    # Find common timestamps
    common_timestamps = set(ground_images.keys()) & set(aerial_images.keys())
    print(f'Total common timestamps: {len(common_timestamps)}')

    timestamps = list(common_timestamps)
    num_timestamps = len(timestamps)

    triplets = []

    for timestamp in timestamps:
        anchor = ground_images[timestamp]
        positive = aerial_images[timestamp]
        # Select a random negative timestamp different from the current one
        negative_timestamp = random.choice(timestamps)
        while negative_timestamp == timestamp:
            negative_timestamp = random.choice(timestamps)
        negative = aerial_images[negative_timestamp]
        triplets.append((anchor, positive, negative))

    print(f'Total triplets: {len(triplets)}')
    return triplets

class TripletDataset(Dataset):
    """
    Custom Dataset for loading triplets of images.
    """
    def __init__(self, triplets, transform_ground=None, transform_aerial=None):
        self.triplets = triplets
        self.transform_ground = transform_ground
        self.transform_aerial = transform_aerial

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]

        # Load images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        # Apply transforms
        if self.transform_ground:
            anchor_img = self.transform_ground(anchor_img)
        if self.transform_aerial:
            positive_img = self.transform_aerial(positive_img)
            negative_img = self.transform_aerial(negative_img)

        return anchor_img, positive_img, negative_img

class EmbeddingNet(nn.Module):
    """
    Embedding network using ResNet-50 as backbone.
    """
    def __init__(self, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        # Load pre-trained ResNet-50
        self.base_model = models.resnet50(pretrained=True)
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

class CrossViewModel(nn.Module):
    """
    Model combining ground and aerial embedding networks.
    """
    def __init__(self, embedding_dim=128):
        super(CrossViewModel, self).__init__()
        self.ground_net = EmbeddingNet(embedding_dim)
        self.aerial_net = EmbeddingNet(embedding_dim)

    def forward(self, anchor_img, positive_img, negative_img):
        # Anchor is a ground-level image
        anchor_embed = self.ground_net(anchor_img)
        # Positive and negative are aerial images
        positive_embed = self.aerial_net(positive_img)
        negative_embed = self.aerial_net(negative_img)
        return anchor_embed, positive_embed, negative_embed

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Cross-View Matching Training Script')
    parser.add_argument('--ground_image_dir', type=str, default='datasets/ground_level_images',
                        help='Directory containing ground-level images.')
    parser.add_argument('--aerial_image_dir', type=str, default='datasets/aerial_image_patches_16',
                        help='Directory containing aerial image patches.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the embedding vector.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints.')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f'Using device: {device}')

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Prepare data
    triplets = create_triplets(args.ground_image_dir, args.aerial_image_dir)

    # Split data into training and validation sets (e.g., 90% training, 10% validation)
    num_triplets = len(triplets)
    split_index = int(0.9 * num_triplets)
    train_triplets = triplets[:split_index]
    val_triplets = triplets[split_index:]

    print(f'Training triplets: {len(train_triplets)}, Validation triplets: {len(val_triplets)}')

    # Define transformations with data augmentation
    transform_ground = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_aerial = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = TripletDataset(train_triplets, transform_ground=transform_ground, transform_aerial=transform_aerial)
    val_dataset = TripletDataset(val_triplets, transform_ground=transform_ground, transform_aerial=transform_aerial)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Initialize model
    model = CrossViewModel(embedding_dim=args.embedding_dim)
    if torch.cuda.device_count() > 1 and args.use_gpu:
        print(f'Using {torch.cuda.device_count()} GPUs.')
        model = nn.DataParallel(model)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        # Train
        model.train()
        total_train_loss = 0.0
        for batch_idx, (anchor_img, positive_img, negative_img) in enumerate(train_loader):
            # Move data to device
            anchor_img = anchor_img.to(device, non_blocking=True)
            positive_img = positive_img.to(device, non_blocking=True)
            negative_img = negative_img.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            anchor_embed, positive_embed, negative_embed = model(anchor_img, positive_img, negative_img)

            # Compute loss
            loss = criterion(anchor_embed, positive_embed, negative_embed)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        average_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args.num_epochs}] Training Loss: {average_train_loss:.4f}')

        # Validate
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for anchor_img, positive_img, negative_img in val_loader:
                anchor_img = anchor_img.to(device, non_blocking=True)
                positive_img = positive_img.to(device, non_blocking=True)
                negative_img = negative_img.to(device, non_blocking=True)

                anchor_embed, positive_embed, negative_embed = model(anchor_img, positive_img, negative_img)
                loss = criterion(anchor_embed, positive_embed, negative_embed)
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{args.num_epochs}] Validation Loss: {average_val_loss:.4f}')

        # Save the model checkpoint if validation loss has decreased
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            checkpoint_path = os.path.join(args.save_dir, f'best_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved best model to {checkpoint_path}')

        # Step the scheduler
        scheduler.step()

    print('Training completed.')

if __name__ == '__main__':
    main()
