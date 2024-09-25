import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image, scale_factor=0.5):
    """Preprocess the image to improve feature detection."""
    # Resize the image
    image_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    # Convert to grayscale (if not already)
    if len(image_resized.shape) > 2:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_preprocessed = clahe.apply(image_resized)
    return image_preprocessed

def match_feature_points(ground_image_path, aerial_image_path, feature_detector='SIFT', max_matches=50):
    # Read the ground-level and aerial images
    ground_image = cv2.imread(ground_image_path)
    aerial_image = cv2.imread(aerial_image_path)

    # Check if images are loaded correctly
    if ground_image is None or aerial_image is None:
        print("Error: Could not load one of the images.")
        return

    # Preprocess images
    ground_image_preprocessed = preprocess_image(ground_image, scale_factor=1.0)  # Keep original scale
    aerial_image_preprocessed = preprocess_image(aerial_image, scale_factor=0.2)  # Scale down for better matching

    # Initialize the feature detector and descriptor extractor
    if feature_detector == 'ORB':
        detector = cv2.ORB_create(nfeatures=5000)  # Increase the number of features to detect
    elif feature_detector == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=5000)
    elif feature_detector == 'SURF':
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    else:
        raise ValueError("Unsupported feature detector. Choose 'ORB', 'SIFT', or 'SURF'.")

    # Detect keypoints and compute descriptors
    kp1, des1 = detector.detectAndCompute(ground_image_preprocessed, None)
    kp2, des2 = detector.detectAndCompute(aerial_image_preprocessed, None)

    # Check if descriptors are None (no keypoints detected)
    if des1 is None or des2 is None:
        print(f"Error: No keypoints detected in one of the images. Ground keypoints: {len(kp1) if kp1 else 0}, Aerial keypoints: {len(kp2) if kp2 else 0}")
        return

    # Ensure that both descriptors have the same type
    if des1.dtype != des2.dtype:
        print(f"Descriptor types do not match. Converting to {des1.dtype}.")
        des2 = des2.astype(des1.dtype)

    # Use BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING if feature_detector == 'ORB' else cv2.NORM_L2, crossCheck=True)
    
    # Match descriptors
    try:
        matches = bf.match(des1, des2)
    except cv2.error as e:
        print(f"Error matching descriptors: {e}")
        return

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the top matches
    matched_image = cv2.drawMatches(ground_image_preprocessed, kp1, aerial_image_preprocessed, kp2, matches[:max_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matching results
    plt.figure(figsize=(12, 6))
    plt.imshow(matched_image, cmap='gray')
    plt.title(f'Top {max_matches} Feature Matches Using {feature_detector}')
    plt.axis('off')
    # plt.show()
    plt.savefig('matched_image_SIFT.png')

# Example usage
ground_image_path = 'datasets/ground_level_images/frame_1723659194.099999.jpg'
# aerial_image_path = 'datasets/aerial_image_patches_64/cropped_frame_1723659171.843820.jpg'
aerial_image_path = 'datasets/ground_level_images/frame_1723659172.200000.jpg'
match_feature_points(ground_image_path, aerial_image_path, feature_detector='SIFT', max_matches=50)
