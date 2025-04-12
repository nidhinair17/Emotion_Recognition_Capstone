import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import openpyxl

class EmotionFeatureExtractor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def extract_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        # Get the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face

        face_roi = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        features = {}

        features['face_x'] = x / frame.shape[1]  
        features['face_y'] = y / frame.shape[0]  
        features['face_width'] = w / frame.shape[1]
        features['face_height'] = h / frame.shape[0]
        features['face_aspect_ratio'] = w / h

        eyes = self.eye_cascade.detectMultiScale(face_roi)
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda x: x[0])  
            left_eye = eyes[0]
            right_eye = eyes[1]

            # Eye measurements
            features['left_eye_x'] = left_eye[0] / w
            features['left_eye_y'] = left_eye[1] / h
            features['left_eye_width'] = left_eye[2] / w
            features['left_eye_height'] = left_eye[3] / h

            features['right_eye_x'] = right_eye[0] / w
            features['right_eye_y'] = right_eye[1] / h
            features['right_eye_width'] = right_eye[2] / w
            features['right_eye_height'] = right_eye[3] / h

            # Eye separation
            eye_separation = abs((left_eye[0] + left_eye[2]/2) - (right_eye[0] + right_eye[2]/2))
            features['eye_separation'] = eye_separation / w
        else:
            # Fill with zeros if eyes not detected
            eye_features = ['left_eye_x', 'left_eye_y', 'left_eye_width', 'left_eye_height',
                          'right_eye_x', 'right_eye_y', 'right_eye_width', 'right_eye_height',
                          'eye_separation']
            for feat in eye_features:
                features[feat] = 0

        # 3. Mouth detection and measurements
        mouth = self.mouth_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.7,
            minNeighbors=11,
            minSize=(25, 15)
        )

        if len(mouth) > 0:
            mouth = max(mouth, key=lambda x: x[2] * x[3])
            features['mouth_x'] = mouth[0] / w
            features['mouth_y'] = mouth[1] / h
            features['mouth_width'] = mouth[2] / w
            features['mouth_height'] = mouth[3] / h
            features['mouth_aspect_ratio'] = mouth[2] / mouth[3]
        else:
            mouth_features = ['mouth_x', 'mouth_y', 'mouth_width', 'mouth_height', 'mouth_aspect_ratio']
            for feat in mouth_features:
                features[feat] = 0

        # 4. Intensity features
        face_roi_resized = cv2.resize(face_roi, (64, 64))
        features['avg_intensity'] = np.mean(face_roi_resized)
        features['intensity_variance'] = np.var(face_roi_resized)

        # Convert features dictionary to list in fixed order
        feature_list = [
            features['face_x'], features['face_y'], features['face_width'], features['face_height'],
            features['face_aspect_ratio'], features['left_eye_x'], features['left_eye_y'],
            features['left_eye_width'], features['left_eye_height'], features['right_eye_x'],
            features['right_eye_y'], features['right_eye_width'], features['right_eye_height'],
            features['eye_separation'], features['mouth_x'], features['mouth_y'],
            features['mouth_width'], features['mouth_height'], features['mouth_aspect_ratio'],
            features['avg_intensity'], features['intensity_variance']
        ]

        return feature_list

def process_video(video_path, extractor):
    """Process video and extract emotion features"""
    cap = cv2.VideoCapture(video_path)
    frame_features = []
    frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        features = extractor.extract_features(frame)
        if features:
            frame_features.append({
                'video_name': os.path.basename(video_path),
                'frame': frame_count,
                'features': features
            })
        frame_count += 1

    cap.release()
    return frame_features

def create_feature_columns():
    """Create column names for emotion features"""
    columns = ['video_name', 'frame']

    feature_names = [
        'face_x', 'face_y', 'face_width', 'face_height', 'face_aspect_ratio',
        'left_eye_x', 'left_eye_y', 'left_eye_width', 'left_eye_height',
        'right_eye_x', 'right_eye_y', 'right_eye_width', 'right_eye_height',
        'eye_separation', 'mouth_x', 'mouth_y', 'mouth_width', 'mouth_height',
        'mouth_aspect_ratio', 'avg_intensity', 'intensity_variance'
    ]

    columns.extend(feature_names)
    return columns

def split_dataset(input_path, train_ratio=0.8):
    input_path = os.path.expanduser(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    video_paths = []
    valid_extensions = ('.flv', '.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')

    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(valid_extensions):
                video_paths.append(os.path.join(root, file))

    if not video_paths:
        print(f"\nCurrent directory being searched: {os.path.abspath(input_path)}")
        raise ValueError(f"No video files found in {input_path}")

    print(f"Found {len(video_paths)} video files")
    return train_test_split(video_paths, train_size=train_ratio, random_state=42)

def save_features_to_excel(features, output_path, dataset_type):
    """Save features to Excel file with summary"""
    columns = create_feature_columns()
    
    # Create DataFrame
    df = pd.DataFrame(columns=columns)
    for item in features:
        row = [item['video_name'], item['frame']] + item['features']
        df.loc[len(df)] = row
    
    # Create timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(output_path, f'{dataset_type}_features_{timestamp}.xlsx')
    
    # Save to Excel with summary
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Features', index=False)
        
        # Add summary sheet
        summary = {
            'Metric': ['Total Videos', 'Total Frames', 'Features per Frame'],
            'Value': [
                df['video_name'].nunique(),
                len(df),
                len(df.columns) - 2
            ]
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)
    
    return excel_path

def main():
    INPUT_PATH = os.path.expanduser("/Users/smriti/Downloads/RAVDESS-Dataset-processed/Angry")
    BASE_OUTPUT_PATH = os.path.expanduser("/Users/smriti/Desktop/Extracted feature files")
    
    # Create separate directories for train and test data
    TRAIN_OUTPUT_PATH = os.path.join(BASE_OUTPUT_PATH, "train")
    TEST_OUTPUT_PATH = os.path.join(BASE_OUTPUT_PATH, "test")
    
    # Create output directories
    os.makedirs(TRAIN_OUTPUT_PATH, exist_ok=True)
    os.makedirs(TEST_OUTPUT_PATH, exist_ok=True)

    print(f"Input path: {os.path.abspath(INPUT_PATH)}")
    print(f"Train output path: {os.path.abspath(TRAIN_OUTPUT_PATH)}")
    print(f"Test output path: {os.path.abspath(TEST_OUTPUT_PATH)}")

    try:
        extractor = EmotionFeatureExtractor()

        print("\nSplitting dataset into train and test sets...")
        train_videos, test_videos = split_dataset(INPUT_PATH)

        print(f"\nNumber of training videos: {len(train_videos)}")
        print(f"Number of test videos: {len(test_videos)}")

        print("\nProcessing training videos...")
        train_features = []
        for video_path in train_videos:
            train_features.extend(process_video(video_path, extractor))

        print("\nProcessing test videos...")
        test_features = []
        for video_path in test_videos:
            test_features.extend(process_video(video_path, extractor))

        print("\nSaving training features...")
        train_file = save_features_to_excel(train_features, TRAIN_OUTPUT_PATH, "train")
        print(f"Training features saved to: {train_file}")

        print("\nSaving test features...")
        test_file = save_features_to_excel(test_features, TEST_OUTPUT_PATH, "test")
        print(f"Test features saved to: {test_file}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure your input directory contains video files")
        print("2. Check that the video files have extensions: .mp4, .avi, or .mov")
        print("3. Verify the input path is correct")
        return

if __name__ == "__main__":
    main()
