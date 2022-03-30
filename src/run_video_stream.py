import settings as s
import os
os.environ['CUDA_VISIBLE_DEVICES']='5'
from tqdm import tqdm
import shutil
from video import video_preprocessing as video
from video import action_recognition as action
import pandas as pd
from datetime import datetime, timedelta
from tinydb import TinyDB
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer
from tinydb.storages import JSONStorage
import torch


#hlvu_location = s.HLVU_LOCATION
#video_path = f"{hlvu_location}/keyframes/shot_split_video"
#data_loc = s.DIR_PATH

#video.process_data(video_path)

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
    splits = shot_name.replace(".mp4", "").split("_")
    shot_num = int(splits[1])
    start_frame, end_frame = df_shots[0][shot_num], df_shots[1][shot_num]
    if start_frame == 0:
        shot_starttime = scene_start_time
    else:
        start_stamp = df_frames.iloc[[start_frame-1]]["Timecode"].tolist()[0]
        start_t = datetime.strptime(start_stamp, '%H:%M:%S.%f')
        start_delta = timedelta(hours=start_t.hour, minutes=start_t.minute, seconds=start_t.second)
        shot_starttime = scene_start_time + start_delta
    end_stamp = df_frames.iloc[[end_frame-1]]["Timecode"].tolist()[0]
    end_t = datetime.strptime(end_stamp, '%H:%M:%S.%f')
    end_delta = timedelta(hours=end_t.hour, minutes=end_t.minute, seconds=end_t.second)
    shot_endtime = scene_start_time + end_delta
    return shot_starttime, shot_endtime





#movie = "honey"

def video_stream(video_path, data_loc, hlvu_location):

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

    # Pick a pretrained model and load the pretrained weights
    model_name = "slowfast_r50"
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)

    #action_frame = {'shot_name': [], 'action': [], 'start_time': [], 'end_time': [], 'scene_name': []}
    #video_df = pd.DataFrame(data=action_frame)
    serialization = SerializationMiddleware(JSONStorage)
    serialization.register_serializer(DateTimeSerializer(), 'TinyDate')
    action_db = TinyDB(f'database/action.json', storage=serialization)

    torch.cuda.set_device(0)
    for scene in tqdm(os.listdir(video_path)):
        #if one_movie:
            #if movie in scene:
        for split in os.listdir(f"{video_path}/{scene}"):
            try:
                action_list = action.get_action_from_video(
                    model,
                    f"{video_path}/{scene}/{split}", 
                    num_frames, 
                    mean, 
                    std, 
                    side_size, 
                    crop_size, 
                    alpha, 
                    sampling_rate, 
                    frames_per_second,
                    data_loc
                )
                movie = scene.partition("-")[0]
                start_time, end_time = get_timestamp(movie, scene, hlvu_location, split)
                #video_df.loc[video_df.shape[0]] = [split, action_list[0], start_time, end_time, scene]
                action_db.insert(
                    {'shot_name': split, 'action': action_list[0], 'start_time': start_time, 'end_time': end_time, 'scene': scene})
            
            except:
                pass


