import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

def extract_and_normalize_landmarks(dataset_path, output_csv_path):
    """Extracts, normalizes, and saves hand landmarks from images to a CSV file."""
    
    # Initialize MediaPipe Hands.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    landmark_data = []
    header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]

    print(f"Starting landmark extraction from: {dataset_path}")
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        return

    # Iterate through each class folder (e.g., 'A', 'B', 'C').
    for label in sorted(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, label)
        if not os.path.isdir(class_path):
            continue

        print(f"Processing class: {label}")

        # Process each image within the class folder.
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Convert BGR image to RGB for MediaPipe.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Convert landmarks to a NumPy array.
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                # Normalize coordinates relative to the wrist.
                base_point = landmarks[0].copy()
                relative_landmarks = landmarks - base_point
                
                # Normalize scale based on the maximum absolute distance.
                max_abs_val = np.max(np.abs(relative_landmarks))
                if max_abs_val > 0:
                    normalized_landmarks = relative_landmarks / max_abs_val
                else:
                    normalized_landmarks = relative_landmarks
                
                # Flatten the normalized landmarks into a single feature vector.
                feature_vector = normalized_landmarks.flatten().tolist()
                
                landmark_data.append([label] + feature_vector)

    # Create a DataFrame and save it to a CSV file.
    df = pd.DataFrame(landmark_data, columns=header)
    df.to_csv(output_csv_path, index=False)
    print(f"\nLandmark extraction complete. Data for {len(df)} images saved to {output_csv_path}")

    hands.close()

if __name__ == '__main__':
    # Define relative paths for dataset and output CSV.
    train_dataset_path = os.path.join('..', 'dataset', 'train')
    output_csv_path = os.path.join('..', 'dataset', 'asl_landmarks.csv')

    # Run the main extraction function.
    extract_and_normalize_landmarks(train_dataset_path, output_csv_path)
