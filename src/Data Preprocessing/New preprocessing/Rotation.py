import cv2
import numpy as np
import os
import random
from tqdm import tqdm

def rotate_video(input_path, angle_range=(-10, 10)):
    """
    Apply rotation augmentation to a video and save as MP4 in the same folder
    
    Args:
        input_path: Path to the input video
        angle_range: Range of rotation angles (min, max) in degrees
        
    Returns:
        Path to the output rotated video
    """
    # Random angle within the specified range
    angle = random.uniform(angle_range[0], angle_range[1])
    
    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output path in the same folder
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    name = os.path.splitext(input_filename)[0]
    output_path = os.path.join(input_dir, f"{name}-rotation.mp4")
    
    # Create video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Rotation parameters
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Process each frame
    for _ in tqdm(range(frame_count), desc=f"Rotating video by {angle:.2f}°"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply rotation
        rotated_frame = cv2.warpAffine(
            frame, 
            rotation_matrix, 
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Write the rotated frame
        out.write(rotated_frame)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Rotated video saved to: {output_path}")
    return output_path

def process_video_folder(folder_path, angle_range=(-10, 10)):
    """
    Process all videos in a folder with rotation augmentation
    
    Args:
        folder_path: Folder containing videos to augment
        angle_range: Range of rotation angles (min, max) in degrees
    """
    # Get all video files
    video_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) 
                  and not f.endswith('-rotation.mp4')]  # Skip already augmented files
    
    # Process each video
    for video_file in video_files:
        input_path = os.path.join(folder_path, video_file)
        rotate_video(input_path, angle_range)

# Example usage
if __name__ == "__main__":
    videos_folder = "/Users/smriti/Downloads/Ravdess"  # Change this to your videos folder
    process_video_folder(videos_folder)