import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import pickle
import cv2
import mediapipe as mp
import soxr
import librosa
import librosa.display
from moviepy.video.io.VideoFileClip import VideoFileClip
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon="😀",
    layout="wide"
)

# Define MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define key landmark indices for emotion-relevant features
LEFT_EYE = [33, 133, 160, 159, 158, 144, 153, 154, 155, 173, 157, 163]
RIGHT_EYE = [362, 385, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381]
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [336, 296, 334, 293, 300]
MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
NOSE = [1, 2, 3, 4, 5, 6, 168, 197, 195, 5, 4, 98, 97, 2, 326, 327]

# Define emotion labels
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Neural Network Models
class CNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=8)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=4, padding=2)
        self.dropout1 = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=8)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=4, padding=2)
        self.dropout2 = nn.Dropout(0.5)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, padding=2)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

class EmotionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=8, dropout_rate=0.3):
        super(EmotionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.bn = nn.BatchNorm1d(hidden_size*2)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        if lstm_out.size(1) == 1:
            lstm_out = lstm_out.squeeze(1)
        else:
            lstm_out = lstm_out[:, -1, :]
        
        lstm_out = self.bn(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class VotingEnsemble(nn.Module):
    def __init__(self, speech_model, facial_model):
        super(VotingEnsemble, self).__init__()
        self.speech_model = speech_model
        self.facial_model = facial_model
        
        for param in self.speech_model.parameters():
            param.requires_grad = False
        for param in self.facial_model.parameters():
            param.requires_grad = False
            
        self.speech_weight = nn.Parameter(torch.tensor(0.5))
        self.facial_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, speech_input, facial_input):
        speech_out = self.speech_model(speech_input)
        facial_out = self.facial_model(facial_input)
        
        speech_probs = torch.softmax(speech_out, dim=1) * torch.sigmoid(self.speech_weight)
        facial_probs = torch.softmax(facial_out, dim=1) * torch.sigmoid(self.facial_weight)
        weighted_avg = (speech_probs + facial_probs) / (torch.sigmoid(self.speech_weight) + torch.sigmoid(self.facial_weight))
        
        return weighted_avg

# Helper functions for feature extraction
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

def extract_landmarks_from_video(video_file, progress_bar=None):
    # Create a temporary file
    # temp_file = NamedTemporaryFile(delete=False, suffix='.mp4')
    # temp_file.write(video_file.read())
    # temp_file.close()
    
    cap = cv2.VideoCapture(video_file)
    features = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a placeholder for video display
    video_placeholder = st.empty()
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as fm:
        for i in range(frame_count):
            if progress_bar is not None:
                progress_bar.progress((i + 1) / frame_count)
            
            ret, img = cap.read()
            if not ret: 
                break
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = fm.process(img_rgb)
            
            # Draw face mesh on the frame
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=img_rgb,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1))
                
                # Display the frame with landmarks
                if i % 5 == 0:  # Only update display every 5 frames to improve performance
                    video_placeholder.image(img_rgb, channels="RGB", caption=f"Processing frame {i+1}/{frame_count}")
                
                # Extract features
                lm = result.multi_face_landmarks[0]
                points = np.array([[p.x, p.y, p.z] for p in lm.landmark])
                try:
                    features.append(extract_features_from_frame(points))
                except:
                    continue
    
    cap.release()
    os.unlink(video_file)
    
    # Clear the video display placeholder
    video_placeholder.empty()
    
    if not features:
        return None
    
    return summarize_video_features(features)

def extract_features_from_face(video_file, progress_bar=None):
    landmarks = extract_landmarks_from_video(video_file, progress_bar)
    if landmarks is None:
        return None, None
    
    df = [landmarks]
    columns = [
        'left_eye_ar', 'right_eye_ar', 'mouth_ar', 'mouth_openness',
        'left_eyebrow_pos', 'right_eyebrow_pos', 'smile_ratio', 'nose_wrinkle',
        'eye_symmetry', 'eyebrow_symmetry'
    ]
    stats = ['mean', 'std', 'min', 'max', 'range', 'delta']
    colnames = [f"{f}_{s}" for f in columns for s in stats]
    feats = pd.DataFrame(df, columns=colnames)

    face_features = feats.values
    
    # Load the scaler
    scaler_path = "C:/Nini/Capstone/CSV_Files/Facial data/New Facial Data/robust_scaler-2.pkl"
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as r:
            scaler = pickle.load(r)
        face_features_scaled = scaler.transform(face_features)
    else:
        # If scaler is not available, just standardize the data
        st.warning("Robust scaler not found. Using standard scaling as a fallback.")
        mean = np.mean(face_features, axis=0)
        std = np.std(face_features, axis=0) + 1e-8
        face_features_scaled = (face_features - mean) / std
    
    tensor_face = torch.tensor(face_features_scaled, dtype=torch.float32).reshape((face_features_scaled.shape[0], 1, face_features_scaled.shape[1]))
    return tensor_face, face_features

# def extract_features_from_audio(video_file, progress_bar=None):
#     # Create a temporary file
#     temp_video = NamedTemporaryFile(delete=False, suffix='.mp4')
#     temp_video.write(video_file.read())
#     temp_video.close()
    
#     temp_audio = NamedTemporaryFile(delete=False, suffix='.wav')
#     temp_audio.close()
    
#     try:
#         # Extract audio from video
#         if progress_bar:
#             progress_bar.progress(0.3)
            
#         video_clip = VideoFileClip(temp_video.name)
#         audio_clip = video_clip.audio
        
#         if audio_clip is None:
#             st.error("No audio track found in the video file.")
#             os.unlink(temp_video.name)
#             os.unlink(temp_audio.name)
#             return None
            
#         audio_clip.write_audiofile(temp_audio.name, verbose=False, logger=None)
        
#         if progress_bar:
#             progress_bar.progress(0.6)
        
#         # Extract audio features
#         X, sample_rate = librosa.load(temp_audio.name, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)
#         audio_resampled = soxr.resample(X, sample_rate, 16000)
#         spectrogram = librosa.feature.melspectrogram(y=audio_resampled, sr=16000, n_mels=128, fmax=8000)
#         db_spec = librosa.power_to_db(spectrogram)
#         log_spectrogram = np.mean(db_spec, axis=1)
        
#         # Load mean and std or create placeholder values
#         mean_path = "C:/Nini/Capstone/src/Data Preprocessing/mean.npy"
#         std_path = "C:/Nini/Capstone/src/Data Preprocessing/std.npy"
        
#         if os.path.exists(mean_path) and os.path.exists(std_path):
#             mean = np.load(mean_path)
#             std = np.load(std_path)
#         else:
#             st.warning("Normalization data not found. Using calculated mean and std as fallback.")
#             mean = np.mean(log_spectrogram)
#             std = np.std(log_spectrogram) + 1e-8
        
#         if progress_bar:
#             progress_bar.progress(0.9)
            
#         mean_tensor = torch.tensor(mean).float()
#         std_tensor = torch.tensor(std).float()
#         log_spectrogram = torch.from_numpy(log_spectrogram).float()
#         log_spectrogram = (log_spectrogram - mean_tensor) / std_tensor

#         log_spectrogram = log_spectrogram.unsqueeze(0).unsqueeze(1).float()
        
#         # Create spectrogram image for display
#         fig, ax = plt.subplots(figsize=(10, 4))
#         img = librosa.display.specshow(db_spec, sr=16000, x_axis='time', y_axis='mel', ax=ax)
#         plt.colorbar(img, format='%+2.0f dB')
#         plt.title('Mel Spectrogram')
#         plt.tight_layout()
        
#         # Save spectrogram to a temporary file
#         temp_spec = NamedTemporaryFile(delete=False, suffix='.png')
#         temp_spec.close()
#         plt.savefig(temp_spec.name)
#         plt.close()
        
#         # Clean up temporary files
#         os.unlink(temp_video.name)
#         os.unlink(temp_audio.name)
        
#         # Return the spectrogram tensor and the path to the spectrogram image
#         return log_spectrogram, temp_spec.name
        
#     except Exception as e:
#         st.error(f"Error processing audio: {e}")
        
#         # Clean up temporary files
#         os.unlink(temp_video.name)
#         os.unlink(temp_audio.name)
#         return None

def extract_features_from_audio(video_file, progress_bar=None):
    # Create a temporary file
    temp_video = NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video.write(video_file.read())
    temp_video.close()

    temp_audio = NamedTemporaryFile(delete=False, suffix='.wav')
    temp_audio.close()

    try:
        # Extract audio from video
        if progress_bar:
            progress_bar.progress(0.3)

        video_clip = VideoFileClip(temp_video.name)
        audio_clip = video_clip.audio

        if audio_clip is None:
            st.error("No audio track found in the video file.")
            video_clip.close()  # <<--- ADD THIS
            os.unlink(temp_video.name)
            os.unlink(temp_audio.name)
            return None

        audio_clip.write_audiofile(temp_audio.name, logger=None)

        video_clip.close()  # <<--- VERY IMPORTANT (close the video before deleting)

        if progress_bar:
            progress_bar.progress(0.6)

        # Now continue with audio processing...
        X, sample_rate = librosa.load(temp_audio.name, res_type='kaiser_fast', duration=3, sr=44100, offset=0.5)
        audio_resampled = soxr.resample(X, sample_rate, 16000)
        spectrogram = librosa.feature.melspectrogram(y=audio_resampled, sr=16000, n_mels=128, fmax=8000)
        db_spec = librosa.power_to_db(spectrogram)
        log_spectrogram = np.mean(db_spec, axis=1)

        mean_path = "C:/Nini/Capstone/src/Data Preprocessing/mean.npy"
        std_path = "C:/Nini/Capstone/src/Data Preprocessing/std.npy"

        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = np.load(mean_path)
            std = np.load(std_path)
        else:
            st.warning("Normalization data not found. Using calculated mean and std as fallback.")
            mean = np.mean(log_spectrogram)
            std = np.std(log_spectrogram) + 1e-8

        if progress_bar:
            progress_bar.progress(0.9)

        mean_tensor = torch.tensor(mean).float()
        std_tensor = torch.tensor(std).float()
        log_spectrogram = torch.from_numpy(log_spectrogram).float()
        log_spectrogram = (log_spectrogram - mean_tensor) / std_tensor

        log_spectrogram = log_spectrogram.unsqueeze(0).unsqueeze(1).float()

        # Create spectrogram image
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(db_spec, sr=16000, x_axis='time', y_axis='mel', ax=ax)
        plt.colorbar(img, format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()

        temp_spec = NamedTemporaryFile(delete=False, suffix='.png')
        temp_spec.close()
        plt.savefig(temp_spec.name)
        plt.close()

        # Now after all processing, safely clean temp files
        # os.unlink(temp_video.name)
        os.unlink(temp_audio.name)

        return log_spectrogram, temp_spec.name, temp_video.name

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        try:
            video_clip.close()  # safely close if error happens
        except:
            pass
        os.unlink(temp_video.name)
        os.unlink(temp_audio.name)
        return None


def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths to model files
    speech_model_path = 'C:/Nini/Capstone/Models/DataAugmentation_cnn_model_new_final_1.pth'
    facial_model_path = 'C:/Nini/Capstone/Models/emotion_lstm_model-7.pth'
    label_encoder_path = "C:/Nini/Capstone/src/Model_training/label_encoder.pkl"
    
    # Check if model files exist
    if not (os.path.exists(speech_model_path) and os.path.exists(facial_model_path)):
        st.error("Model files not found. Please make sure they are in the correct directory.")
        st.info("Looking for files: speech_model.pth, facial_model.pth, label_encoder.pkl")
        return None, None, None, None
    
    # Initialize models (with placeholder input sizes that will be adjusted at runtime)
    speech_model = CNNModel(input_size=128, num_classes=8).to(device)
    facial_model = EmotionLSTM(input_size=60).to(device)
    
    # Load pre-trained weights
    speech_model.load_state_dict(torch.load(speech_model_path, map_location=device))
    facial_model.load_state_dict(torch.load(facial_model_path, map_location=device))
    
    # Initialize ensemble model
    ensemble_model = VotingEnsemble(speech_model, facial_model).to(device)
    
    # Load label encoder
    if os.path.exists(label_encoder_path):
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    else:
        st.warning("Label encoder not found. Using default emotion labels.")
        # Create a simple mapping from indices to emotion labels
        label_encoder = {i: label for i, label in enumerate(EMOTION_LABELS)}
    
    return speech_model, facial_model, ensemble_model, label_encoder

def evaluate_individual_vs_ensemble(speech_model, facial_model, ensemble_model, 
                                   speech_data, facial_data, device='cpu'):
    speech_data = speech_data.to(device)
    facial_data = facial_data.to(device)
    
    speech_model.eval()
    facial_model.eval()
    ensemble_model.eval()
    
    with torch.no_grad():
        speech_outputs = speech_model(speech_data)
        facial_outputs = facial_model(facial_data)
        ensemble_outputs = ensemble_model(speech_data, facial_data)
        
        _, speech_preds = torch.max(speech_outputs, 1)
        _, facial_preds = torch.max(facial_outputs, 1)
        _, ensemble_preds = torch.max(ensemble_outputs, 1)
        
        # Get probabilities
        speech_probs = torch.softmax(speech_outputs, dim=1)
        facial_probs = torch.softmax(facial_outputs, dim=1)
        
        speech_preds_np = speech_preds.cpu().numpy()
        facial_preds_np = facial_preds.cpu().numpy()
        ensemble_preds_np = ensemble_preds.cpu().numpy()
        
        speech_probs_np = speech_probs.cpu().numpy()
        facial_probs_np = facial_probs.cpu().numpy()
        ensemble_probs_np = ensemble_outputs.cpu().numpy()

        return speech_preds_np, facial_preds_np, ensemble_preds_np, speech_probs_np, facial_probs_np, ensemble_probs_np

def plot_prediction_bars(probs, labels):
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    
    ax.barh(y_pos, probs[0] * 100, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Probability (%)')
    ax.set_title('Emotion Prediction Probabilities')
    
    # Add percentage labels to bars
    for i, v in enumerate(probs[0]):
        ax.text(v * 100 + 1, i, f'{v*100:.1f}%', va='center')
    
    plt.tight_layout()
    return fig

# def plot_prediction_bars(probs, labels):
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Create a list of (probability, label) pairs and sort by probability (descending)
#     prob_label_pairs = [(prob, label) for prob, label in zip(probs[0], labels)]
    
#     # Extract sorted probabilities and labels
#     sorted_pairs = sorted(prob_label_pairs, reverse=True)
#     sorted_probs = [pair[0] for pair in sorted_pairs]
#     sorted_labels = [pair[1] for pair in sorted_pairs]
    
#     y_pos = np.arange(len(sorted_labels))
    
#     ax.barh(y_pos, np.array(sorted_probs) * 100, align='center')
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(sorted_labels)
#     ax.invert_yaxis()  # Labels read top-to-bottom
#     ax.set_xlabel('Probability (%)')
#     ax.set_title('Emotion Prediction Probabilities')
    
#     # Add percentage labels to bars
#     for i, v in enumerate(sorted_probs):
#         ax.text(v * 100 + 1, i, f'{v*100:.1f}%', va='center')
    
#     plt.tight_layout()
#     return fig

# def plot_prediction_bars(probs, labels, label_encoder=None):
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Debug info
#     print(f"Probabilities shape: {probs.shape}")
#     print(f"Number of labels: {len(labels)}")
    
#     # If we have a label encoder available, use it to ensure correct ordering
#     if label_encoder and hasattr(label_encoder, 'classes_'):
#         ordered_labels = list(label_encoder.classes_)
#         print(f"Label encoder classes: {ordered_labels}")
#     else:
#         ordered_labels = labels
#         print(f"Using provided labels: {ordered_labels}")
    
#     # Ensure we have the right number of probabilities
#     if len(ordered_labels) != probs.shape[1]:
#         print(f"WARNING: Number of labels ({len(ordered_labels)}) doesn't match number of probabilities ({probs.shape[1]})")
    
#     y_pos = np.arange(len(ordered_labels))
    
#     # Create the horizontal bar chart
#     ax.barh(y_pos, probs[0] * 100, align='center')
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(ordered_labels)
#     ax.invert_yaxis()  # Labels read top-to-bottom
#     ax.set_xlabel('Probability (%)')
#     ax.set_title('Emotion Prediction Probabilities')
    
#     # Add percentage labels to bars
#     for i, v in enumerate(probs[0]):
#         ax.text(v * 100 + 1, i, f'{v*100:.1f}%', va='center')
#         # Debug: print which probability goes with which label
#         print(f"Label: {ordered_labels[i]}, Probability: {v*100:.1f}%")
    
#     plt.tight_layout()
#     return fig

def main():
    st.title("Multimodal Emotion Recognition System")
    st.markdown("""
    This application uses deep learning models to recognize emotions from both speech and facial expressions in videos.
    Upload a video file and the system will analyze it to predict the emotional content.
    """)
    
    # Create a sidebar for options and information
    st.sidebar.title("About")
    st.sidebar.info("""
    This application combines speech and facial analysis to detect emotions in videos.
    It uses:
    - CNN for audio processing
    - LSTM for facial expression analysis
    - Ensemble model to combine both modalities
    
    Supported emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
    """)
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Check if models are available
        with st.spinner("Loading models..."):
            speech_model, facial_model, ensemble_model, label_encoder = load_models()
            
        if speech_model is None or facial_model is None:
            st.error("Failed to load models. Please check the model files.")
            return
            
        # Create a copy of the uploaded file for each processing step
        speech_file = uploaded_file
        facial_file = uploaded_file
        
        
        st.subheader("Processing Video")
        
        # Process speech
        with st.spinner("Extracting audio features..."):
            col1, col2 = st.columns(2)
            with col1:
                speech_progress = st.progress(0)
                speech_features_result = extract_features_from_audio(speech_file, speech_progress)
                
                if speech_features_result is None:
                    st.error("Failed to extract audio features from the video.")
                    return
                    
                speech_features, spectrogram_path, facial_file = speech_features_result
                speech_progress.progress(1.0)
                st.success("Audio processing complete!")
                
                # Display the spectrogram
                st.subheader("Audio Spectrogram")
                st.image(spectrogram_path, caption="Mel Spectrogram")
                os.unlink(spectrogram_path)  # Clean up the temporary file
        
        # Process facial expressions
        with st.spinner("Extracting facial features..."):
            with col2:
                face_progress = st.progress(0)
                face_result = extract_features_from_face(facial_file, face_progress)
                
                if face_result is None or face_result[0] is None:
                    st.error("Failed to detect faces in the video. Please ensure the video contains visible faces.")
                    return
                    
                face_features, face_raw = face_result
                face_progress.progress(1.0)
                st.success("Facial processing complete!")
        
        # Make predictions
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with st.spinner("Making predictions..."):
            speech_preds, facial_preds, ensemble_preds, speech_probs, facial_probs, ensemble_probs = evaluate_individual_vs_ensemble(
                speech_model, facial_model, ensemble_model,
                speech_features, face_features, device
            )
            
            # Convert predictions to emotions
            if isinstance(label_encoder, dict):
                speech_emotion = label_encoder[speech_preds[0]]
                facial_emotion = label_encoder[facial_preds[0]]
                ensemble_emotion = label_encoder[ensemble_preds[0]]
            else:
                speech_emotion = label_encoder.inverse_transform(speech_preds)[0]
                facial_emotion = label_encoder.inverse_transform(facial_preds)[0]
                ensemble_emotion = label_encoder.inverse_transform(ensemble_preds)[0]
        
        # Display results
        st.header("Emotion Recognition Results")
        
        # Create 3 columns for the different models
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Speech Analysis")
            st.metric("Predicted Emotion", speech_emotion)
            speech_fig = plot_prediction_bars(speech_probs, EMOTION_LABELS)
            st.pyplot(speech_fig)
            
        with col2:
            st.subheader("Facial Analysis")
            st.metric("Predicted Emotion", facial_emotion)
            facial_fig = plot_prediction_bars(facial_probs, EMOTION_LABELS)
            st.pyplot(facial_fig)
            
        with col3:
            st.subheader("Ensemble (Combined)")
            st.metric("Predicted Emotion", ensemble_emotion, 
                      delta="Final Prediction" if ensemble_emotion == speech_emotion == facial_emotion else None)
            ensemble_fig = plot_prediction_bars(ensemble_probs, EMOTION_LABELS)
            st.pyplot(ensemble_fig)
        
        # Display a visual indicator of the final emotion
        st.header("Final Emotion Classification")
        
        # Create a centered layout for the emoji
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            emotion_icons = {
                "neutral": "😐",
                "calm": "😌",
                "happy": "😄",
                "sad": "😢",
                "angry": "😠",
                "fearful": "😨",
                "disgust": "🤢",
                "surprised": "😲"
            }
            
            st.markdown(f"<h1 style='text-align: center;'>{emotion_icons.get(ensemble_emotion, '❓')} {ensemble_emotion.capitalize()}</h1>", unsafe_allow_html=True)
            
            # Add confidence level
            confidence = float(max(ensemble_probs[0]) * 100)
            st.markdown(f"<h3 style='text-align: center;'>Confidence: {confidence:.1f}%</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()