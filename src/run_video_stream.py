import settings as s
import os
import tqdm as tqdm
import shutil
from video import video_preprocessing as video
from video import action_recognition as action
import pandas as pd
from datetime import datetime, timedelta


hlvu_location = s.HLVU_LOCATION
video_path = f"{hlvu_location}/keyframes/shot_split_video"

video.process_data(video_path)

def get_timestamp(movie, movie_scene, hlvu_location, shot_name):
    path = f"{hlvu_location}/scene.segmentation.reference/{movie}.csv"
    df_scenes = pd.read_csv(path, header=None)
    # load keyframe txt file and csv
    shot_txt = f"{hlvu_location}/keyframes/shot_txt/{movie_scene}.txt"
    shot_csv = f"{hlvu_location}/keyframes/shot_stats/{movie_scene}.csv"
    df_shots = pd.read_csv(shot_txt, header = None, sep = " ")
    df_frames = pd.read_csv(shot_csv,skiprows=1,delimiter=",")
    # calculate scene start time
    scene_ind = int(movie_scene.split("-")[1]) -1
    scene_start_time = datetime.strptime(df_scenes.iloc[[scene_ind]][0].to_list()[0], '%H:%M:%S')
    splits = shot_name.strip(".mp4").split("_")
    shot_num = int(splits[1])
    start_frame, end_frame = df_shots[shot_num][0], df_shots[shot_num][1]
    if start_frame == 0:
        shot_starttime = scene_start_time
    else:
        start_stamp = df_frames.iloc[[start_frame-1]]["Timecode"].tolist()[0]
        end_stamp = df_frames.iloc[[end_frame-1]]["Timecode"].tolist()[0]
        start_t = datetime.strptime(end_stamp, '%H:%M:%S.%f')
        start_delta = timedelta(hours=start_t.hour, minutes=start_t.minute, seconds=start_t.second)
        shot_starttime = scene_start_time + start_delta
    end_stamp = df_frames.iloc[[end_frame-1]]["Timecode"].tolist()[0]
    end_t = datetime.strptime(end_stamp, '%H:%M:%S.%f')
    end_delta = timedelta(hours=end_t.hour, minutes=end_t.minute, seconds=end_t.second)
    shot_endtime = scene_start_time + end_delta
    return shot_starttime.time(), shot_endtime.time()




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

one_movie = True
movie = "honey"

action_frame = {'shot_name': [], 'action': [], 'start_time': [], 'end_time': [], 'scene_name': []}
video_df = pd.DataFrame(data=action_frame)

for scene in os.listdir(video_path):
    if one_movie:
        if movie in scene:
            for split in os.listdir(f"{video_path}/{movie}"):

                action_list = action.get_action_from_video(
                    f"{video_path}/{movie}/{split}", 
                    num_frames, 
                    mean, 
                    std, 
                    side_size, 
                    crop_size, 
                    alpha, 
                    sampling_rate, 
                    frames_per_second
                )

                start_time, end_time = get_timestamp(movie, scene, hlvu_location, split)
                video_df.loc[video_df.shape[0]] = [split, action_list[0], start_time, end_time, scene]

