import os
from moviepy.video.io.VideoFileClip import VideoFileClip

data_folder = '../Data'
dataset_folder = '../AudioFiles'
for folder in os.listdir(data_folder):
    if os.path.isdir(f'{data_folder}/{folder}'):
        folder_path = f'{data_folder}/{folder}'
        dest_folder_path = f'{dataset_folder}/{folder}'
        os.makedirs(dest_folder_path,exist_ok=True)
        for file in os.listdir(folder_path):
            # print(file)
            vid_file = f'{folder_path}/{file}'
            audio_file = f'{dest_folder_path}/{file.split('.')[0]}.wav'
            print(audio_file)
            video_clip = VideoFileClip(vid_file)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_file)
            audio_clip.close()
            video_clip.close()
