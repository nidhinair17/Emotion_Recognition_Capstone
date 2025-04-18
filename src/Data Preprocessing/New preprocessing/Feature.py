# import os
# import cv2
# import numpy as np
# import pandas as pd
# import mediapipe as mp
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # Define key landmark indices for emotion-relevant features
# # These indices are based on MediaPipe's 468 facial landmarks

# # Eyes
# LEFT_EYE = [33, 133, 160, 159, 158, 144, 153, 154, 155, 173, 157, 163]
# RIGHT_EYE = [362, 385, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381]

# # Eyebrows
# LEFT_EYEBROW = [70, 63, 105, 66, 107]
# RIGHT_EYEBROW = [336, 296, 334, 293, 300]

# # Mouth
# MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
# MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# # Nose
# NOSE = [1, 2, 3, 4, 5, 6, 168, 197, 195, 5, 4, 98, 97, 2, 326, 327]

# def extract_landmarks_from_video(video_path):
#     """Extract facial landmarks from each frame of a video."""
#     cap = cv2.VideoCapture(video_path)
#     all_landmarks = []
    
#     with mp_face_mesh.FaceMesh(
#         static_image_mode=False,
#         max_num_faces=1,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as face_mesh:
        
#         while cap.isOpened():
#             success, image = cap.read()
#             if not success:
#                 break
                
#             # Convert to RGB for MediaPipe
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
#             # Process the image
#             results = face_mesh.process(image_rgb)
            
#             if results.multi_face_landmarks:
#                 face_landmarks = results.multi_face_landmarks[0]
                
#                 # Convert landmarks to numpy array
#                 frame_landmarks = np.array([(landmark.x, landmark.y, landmark.z) 
#                                            for landmark in face_landmarks.landmark])
                
#                 all_landmarks.append(frame_landmarks)
        
#     cap.release()
#     return np.array(all_landmarks)

# def calculate_distance(point1, point2):
#     """Calculate Euclidean distance between two points."""
#     return np.sqrt(np.sum((point1 - point2)**2))

# def calculate_eye_aspect_ratio(eye_landmarks):
#     """Calculate eye aspect ratio (height/width)."""
#     # Vertical distances
#     v1 = calculate_distance(eye_landmarks[1], eye_landmarks[7])
#     v2 = calculate_distance(eye_landmarks[2], eye_landmarks[6])
#     v3 = calculate_distance(eye_landmarks[3], eye_landmarks[5])
    
#     # Horizontal distance
#     h = calculate_distance(eye_landmarks[0], eye_landmarks[4])
    
#     # Eye aspect ratio
#     ear = (v1 + v2 + v3) / (3.0 * h)
#     return ear

# def calculate_mouth_aspect_ratio(outer_landmarks, inner_landmarks):
#     """Calculate mouth aspect ratio (height/width)."""
#     # Outer lip vertical distance
#     outer_v = calculate_distance(outer_landmarks[3], outer_landmarks[9])
    
#     # Inner lip vertical distance
#     inner_v = calculate_distance(inner_landmarks[3], inner_landmarks[9])
    
#     # Horizontal distance
#     h = calculate_distance(outer_landmarks[0], outer_landmarks[6])
    
#     # Mouth aspect ratio and openness
#     mar = outer_v / h
#     mouth_openness = inner_v / outer_v
    
#     return mar, mouth_openness

# def calculate_eyebrow_position(eyebrow_landmarks, eye_landmarks):
#     """Calculate eyebrow position relative to eye."""
#     eyebrow_y = np.mean([landmark[1] for landmark in eyebrow_landmarks])
#     eye_y = np.mean([landmark[1] for landmark in eye_landmarks])
    
#     # Lower y-value means higher position on image
#     return eye_y - eyebrow_y

# def extract_emotion_features(landmarks_3d):
#     """Extract emotion-relevant features from facial landmarks."""
#     emotion_features = []
    
#     for frame_landmarks in landmarks_3d:
#         # Extract specific landmark groups
#         left_eye_landmarks = frame_landmarks[LEFT_EYE]
#         right_eye_landmarks = frame_landmarks[RIGHT_EYE]
#         left_eyebrow_landmarks = frame_landmarks[LEFT_EYEBROW]
#         right_eyebrow_landmarks = frame_landmarks[RIGHT_EYEBROW]
#         mouth_outer_landmarks = frame_landmarks[MOUTH_OUTER]
#         mouth_inner_landmarks = frame_landmarks[MOUTH_INNER]
#         nose_landmarks = frame_landmarks[NOSE]
        
#         # Face width and height for normalization
#         face_width = np.max(frame_landmarks[:, 0]) - np.min(frame_landmarks[:, 0])
#         face_height = np.max(frame_landmarks[:, 1]) - np.min(frame_landmarks[:, 1])
        
#         # Calculate features
#         # 1. Eye aspect ratios
#         left_ear = calculate_eye_aspect_ratio(left_eye_landmarks)
#         right_ear = calculate_eye_aspect_ratio(right_eye_landmarks)
        
#         # 2. Mouth measurements
#         mar, mouth_openness = calculate_mouth_aspect_ratio(mouth_outer_landmarks, mouth_inner_landmarks)
        
#         # 3. Eyebrow positions (higher values mean more raised eyebrows)
#         left_eyebrow_position = calculate_eyebrow_position(left_eyebrow_landmarks, left_eye_landmarks)
#         right_eyebrow_position = calculate_eyebrow_position(right_eyebrow_landmarks, right_eye_landmarks)
        
#         # 4. Mouth corners (for smile detection)
#         left_corner = mouth_outer_landmarks[0]
#         right_corner = mouth_outer_landmarks[6]
#         middle_top = mouth_outer_landmarks[3]
#         middle_bottom = mouth_outer_landmarks[9]
        
#         # Normalize mouth corner height relative to mouth center
#         mouth_center_y = (middle_top[1] + middle_bottom[1]) / 2
#         left_corner_rel_y = mouth_center_y - left_corner[1]
#         right_corner_rel_y = mouth_center_y - right_corner[1]
        
#         # Positive values indicate upturned corners (smiling)
#         smile_ratio = (left_corner_rel_y + right_corner_rel_y) / (2 * face_height)
        
#         # 5. Nose wrinkle (for disgust)
#         nose_wrinkle = np.std([landmark[2] for landmark in nose_landmarks])  # Z-coordinate variance
        
#         # 6. Symmetry features (for detecting some emotions like contempt)
#         eye_symmetry = abs(left_ear - right_ear)
#         eyebrow_symmetry = abs(left_eyebrow_position - right_eyebrow_position)
        
#         # Combine all features
#         frame_features = [
#             left_ear, right_ear,                      # Eye openness
#             mar, mouth_openness,                      # Mouth shape
#             left_eyebrow_position/face_height,        # Normalized eyebrow positions
#             right_eyebrow_position/face_height,
#             smile_ratio,                              # Smile detection
#             nose_wrinkle,                             # Disgust indicator
#             eye_symmetry,                             # Facial symmetry
#             eyebrow_symmetry
#         ]
        
#         emotion_features.append(frame_features)
    
#     return np.array(emotion_features)

# def process_ravdess_dataset(dataset_path, excel_output_path="ravdess_emotion_features.xlsx"):
#     """Process the RAVDESS dataset, extracting features and labels."""
#     all_features = []
#     video_paths = []
#     emotions = []
    
#     # RAVDESS filename format: 01-01-01-01-01-01-01.mp4
#     # where the 3rd number is the emotion label:
#     # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 
#     # 06 = fearful, 07 = disgust, 08 = surprised
#     emotion_map = {
#         '01': 'neutral',
#         '02': 'calm',
#         '03': 'happy',
#         '04': 'sad',
#         '05': 'angry',
#         '06': 'fearful',
#         '07': 'disgust',
#         '08': 'surprised'
#     }
    
#     # Walk through all directories in the dataset
#     for root, dirs, files in tqdm(list(os.walk(dataset_path)), desc="Processing directories"):
#         for file in tqdm(files, desc=f"Processing files in {os.path.basename(root)}"):
#             if file.endswith('.mp4'):
#                 # Extract emotion label from filename
#                 parts = file.split('-')
#                 if len(parts) >= 3:
#                     emotion_label = parts[2]
#                     emotion_name = emotion_map.get(emotion_label, 'unknown')
                    
#                     video_path = os.path.join(root, file)
#                     print(f"Processing {file}, emotion: {emotion_name}")
                    
#                     try:
#                         # Extract landmarks from video
#                         landmarks = extract_landmarks_from_video(video_path)
                        
#                         if len(landmarks) > 0:
#                             # Extract emotion features from landmarks
#                             features = extract_emotion_features(landmarks)
                            
#                             # Aggregate features over the video (mean for now)
#                             video_features = np.mean(features, axis=0)
                            
#                             all_features.append(video_features)
#                             video_paths.append(video_path)
#                             emotions.append(emotion_name)
                            
#                     except Exception as e:
#                         print(f"Error processing {file}: {str(e)}")
    
#     # Create DataFrame
#     df = pd.DataFrame(all_features)
#     df.columns = [
#         'left_eye_ar', 'right_eye_ar', 
#         'mouth_ar', 'mouth_openness', 
#         'left_eyebrow_pos', 'right_eyebrow_pos',
#         'smile_ratio', 'nose_wrinkle',
#         'eye_symmetry', 'eyebrow_symmetry'
#     ]
    
#     # Add metadata
#     df['video_path'] = video_paths
#     df['emotion'] = emotions
    
#     # Save to Excel
#     df.to_excel(excel_output_path, index=False)
#     print(f"Features saved to {excel_output_path}")
    
#     return df

# def create_train_test_split(df, test_size=0.2, random_state=42):
#     train_df, test_df = train_test_split(
#         df, 
#         test_size=test_size, 
#         random_state=random_state,
#         stratify=df['emotion']  # Ensure balanced emotions in both sets
#     )
    
#     train_df.to_excel("/Users/smriti/Desktop/train.xlsx", index=False)
#     test_df.to_excel("/Users/smriti/Desktop/test.xlsx", index=False)
    
#     print(f"Train set: {len(train_df)} samples")
#     print(f"Test set: {len(test_df)} samples")
    
#     # Display class distribution
#     print("\nTrain set class distribution:")
#     print(train_df['emotion'].value_counts())
    
#     print("\nTest set class distribution:")
#     print(test_df['emotion'].value_counts())
    
#     return train_df, test_df

# def main():
#     dataset_path = "/Users/smriti/Downloads/Ravdess"  
    
#     print("Processing RAVDESS dataset...")
#     df = process_ravdess_dataset(dataset_path)
    
#     print("\nCreating train/test split...")
#     train_df, test_df = create_train_test_split(df, test_size=0.2)
    
    
#     print("\nData processing complete! You now have:")
#     print("1. Raw features in ravdess_emotion_features.xlsx")
#     print("2. Train/test split in train_data.xlsx and test_data.xlsx")

# if __name__ == "__main__":
#     main()


import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define key landmark indices for emotion-relevant features
LEFT_EYE = [33, 133, 160, 159, 158, 144, 153, 154, 155, 173, 157, 163]
RIGHT_EYE = [362, 385, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
NOSE = [1, 2, 3, 4, 5, 6, 168, 197, 195, 5, 4, 98, 97, 2, 326, 327]

def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def calculate_eye_aspect_ratio(eye):
    v1 = calculate_distance(eye[1], eye[7])
    v2 = calculate_distance(eye[2], eye[6])
    v3 = calculate_distance(eye[3], eye[5])
    h = calculate_distance(eye[0], eye[4])
    return (v1 + v2 + v3) / (3.0 * h)

def calculate_mouth_aspect_ratio(outer, inner):
    outer_v = calculate_distance(outer[3], outer[9])
    inner_v = calculate_distance(inner[3], inner[9])
    h = calculate_distance(outer[0], outer[6])
    return outer_v / h, inner_v / outer_v

def calculate_eyebrow_position(eyebrow, eye):
    return np.mean([p[1] for p in eye]) - np.mean([p[1] for p in eyebrow])

def extract_features_from_frame(frame):
    left_eye = frame[LEFT_EYE]
    right_eye = frame[RIGHT_EYE]
    left_eyebrow = frame[LEFT_EYEBROW]
    right_eyebrow = frame[RIGHT_EYEBROW]
    mouth_outer = frame[MOUTH_OUTER]
    mouth_inner = frame[MOUTH_INNER]
    nose = frame[NOSE]

    width = np.max(frame[:, 0]) - np.min(frame[:, 0])
    height = np.max(frame[:, 1]) - np.min(frame[:, 1])

    left_ear = calculate_eye_aspect_ratio(left_eye)
    right_ear = calculate_eye_aspect_ratio(right_eye)
    mar, openness = calculate_mouth_aspect_ratio(mouth_outer, mouth_inner)
    l_eyebrow_pos = calculate_eyebrow_position(left_eyebrow, left_eye) / height
    r_eyebrow_pos = calculate_eyebrow_position(right_eyebrow, right_eye) / height
    mouth_center_y = (mouth_outer[3][1] + mouth_outer[9][1]) / 2
    smile = ((mouth_center_y - mouth_outer[0][1]) + (mouth_center_y - mouth_outer[6][1])) / (2 * height)
    nose_wrinkle = np.std([p[2] for p in nose])
    eye_sym = abs(left_ear - right_ear)
    brow_sym = abs(l_eyebrow_pos - r_eyebrow_pos)

    return [left_ear, right_ear, mar, openness, l_eyebrow_pos, r_eyebrow_pos, smile, nose_wrinkle, eye_sym, brow_sym]

def summarize_video_features(features):
    features = np.array(features)
    summary = []
    for i in range(features.shape[1]):
        f = features[:, i]
        summary.extend([np.mean(f), np.std(f), np.min(f), np.max(f), np.max(f)-np.min(f), f[-1] - f[0]])
    return summary

def extract_landmarks_from_video(path):
    cap = cv2.VideoCapture(path)
    features = []
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as fm:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret: break
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = fm.process(img_rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0]
                points = np.array([[p.x, p.y, p.z] for p in lm.landmark])
                try:
                    features.append(extract_features_from_frame(points))
                except:
                    continue
    cap.release()
    return summarize_video_features(features)

def process_dataset(path):
    df = []
    emo_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".mp4"):
                emo_id = f.split("-")[2]
                label = emo_map.get(emo_id, 'unknown')
                file_path = os.path.join(root, f)
                print(f"Processing {f} → {label}")
                try:
                    feats = extract_landmarks_from_video(file_path)
                    df.append(feats + [file_path, label])
                except:
                    print(f"Error in {f}")
    columns = [
        'left_eye_ar', 'right_eye_ar', 'mouth_ar', 'mouth_openness',
        'left_eyebrow_pos', 'right_eyebrow_pos', 'smile_ratio', 'nose_wrinkle',
        'eye_symmetry', 'eyebrow_symmetry'
    ]
    stats = ['mean', 'std', 'min', 'max', 'range', 'delta']
    colnames = [f"{f}_{s}" for f in columns for s in stats] + ['video_path', 'emotion']
    return pd.DataFrame(df, columns=colnames)

def main():
    dataset_path = "/Users/smriti/Downloads/Ravdess"
    df = process_dataset(dataset_path)
    df.to_excel("/Users/smriti/Downloads/Ravdess/ravdess_temporal_features.xlsx", index=False)
    print("Saved to ravdess_temporal_features.xlsx")

if __name__ == "__main__":
    main()
