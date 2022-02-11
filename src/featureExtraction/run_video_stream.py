import settings as s
import os
import tqdm as tqdm
import shutil
from video import video_preprocessing as video
from video import action_recognition as action

hlvu_location = s.HLVU_LOCATION
video_path = f"{hlvu_location}/keyframes/shot_split_video"

video.process_data(video_path)

####################
# SlowFast transform
####################

num_frames = 32
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
side_size = 256
crop_size = 256
alpha = 4
sampling_rate = 2
frames_per_second = 30

action.get_action_from_video(
    video_path, 
    num_frames, 
    mean, 
    std, 
    side_size, 
    crop_size, 
    alpha, 
    sampling_rate, 
    frames_per_second
)
