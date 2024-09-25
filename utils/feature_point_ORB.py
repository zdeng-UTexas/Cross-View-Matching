import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_feature_points(ground_image_path, aerial_image_path, feature_detector='ORB', max_matches=50):
    # Read the ground-level and aerial images
    ground_image = cv2.imread(ground_image_path, cv2.IMREAD_GRAYSCALE)
    aerial_image = cv2.imread(aerial_image_path, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded correctly
    if ground_image is None or aerial_image is None:
        print("Error: Could not load one of the images.")
        return

    # Initialize the feature detector and descriptor extractor
    if feature_detector == 'ORB':
        detector = cv2.ORB_create()
    elif feature_detector == 'SIFT':
        detector = cv2.SIFT_create()
    elif feature_detector == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    else:
        raise ValueError("Unsupported feature detector. Choose 'ORB', 'SIFT', or 'SURF'.")

    # Detect keypoints and compute descriptors
    kp1, des1 = detector.detectAndCompute(ground_image, None)
    kp2, des2 = detector.detectAndCompute(aerial_image, None)

    # Check if descriptors are None (no keypoints detected)
    if des1 is None or des2 is None:
        print("Error: No keypoints detected in one of the images.")
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
    matched_image = cv2.drawMatches(ground_image, kp1, aerial_image, kp2, matches[:max_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matching results
    plt.figure(figsize=(12, 6))
    plt.imshow(matched_image)
    plt.title(f'Top {max_matches} Feature Matches Using {feature_detector}')
    plt.axis('off')
    plt.savefig('matched_image_ORB.png')
    # plt.show()

# Example usage
# ground_image_path = 'datasets/ground_level_images/frame_1723659171.843820.jpg'
ground_image_path = 'datasets/ground_level_images/frame_1723659194.099999.jpg'
aerial_image_path = 'datasets/aerial_image_patches_64/cropped_frame_1723659171.843820.jpg'
# aerial_image_path = 'datasets/ground_level_images/frame_1723659194.099999.jpg'
# aerial_image_path = 'datasets/ground_level_images/frame_1723659172.200000.jpg'
match_feature_points(ground_image_path, aerial_image_path, feature_detector='ORB', max_matches=50)
