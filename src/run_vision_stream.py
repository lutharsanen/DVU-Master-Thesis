import sys, os
#os.environ["CUDA_VISIBLE_DEVICES"]="4"

from sklearn import cluster
from tqdm import tqdm
import os
from vision import evaluation, training
from vision import compare
from vision import data_creation
import settings as s
import pandas as pd
import shutil
import vision.places365.run_placesCNN_basic as places365 
import tensorflow as tf
from datetime import datetime, timedelta
from tinydb import TinyDB
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer
from tinydb.storages import JSONStorage
import torch
import tensorflow_hub as hub

#hlvu_location = s.HLVU_LOCATION
#dir_path = s.DIR_PATH
#img_path = f"{hlvu_location}/keyframes/shot_keyf"

def get_timestamp_from_shot(movie, movie_scene, shot_name, hlvu_location):
    # load scene segmentation csv
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
    # calculate keyframe timestamp
    splits = shot_name.strip(".jpg").split("_")
    shot_num = int(splits[1])
    shot_ind = int(splits[3])+2
    #print(shot_num, shot_ind)
    frame_ind = df_shots[shot_ind][shot_num] -1
    timestamp = df_frames.iloc[[frame_ind]]["Timecode"].to_list()[0]
    t = datetime.strptime(timestamp, '%H:%M:%S.%f')
    delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    # calculate actual frame time stamp
    new_time = scene_start_time + delta
    # return timestamp
    #print(new_time.time())
    return new_time

#movie_list = ["shooters", "The_Big_Something", "time_expired", "Valkaama", "Huckleberry_Finn", "spiritual_contact", "honey", "sophie", "Nuclear_Family", "SuperHero"]


def run(movie_list, hlvu_location, dir_path, img_path, testset = False):

    #movies = "honey"
    #preload delf model
    delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

    #for movies in os.listdir(img_path):
    for movies in movie_list:
        serialization = SerializationMiddleware(JSONStorage)
        serialization.register_serializer(DateTimeSerializer(), 'TinyDate')
        vision_db = TinyDB(f'database/vision_{movies}.json', storage=serialization)
        scenelist = os.listdir(f"{img_path}/{movies}")
        orderedshotlist = [i.partition('-')[2] for i in scenelist]

        # finetuning face recognition model
        for i in tqdm(range(len(scenelist))):
            num = i+1
            list_index = orderedshotlist.index(str(num))
            shots = scenelist[list_index]
            for shot in os.listdir(f"{img_path}/{movies}/{shots}"):
                path = f"{img_path}/{movies}/{shots}/{shot}"
                if testset:
                    training(path, movies, hlvu_location, testset = True)
                else:
                    training(path, movies, hlvu_location)
        
        # delete existing model to generate new one
        if testset:
            model_loc = f"{hlvu_location}/Queries/movie_knowledge_graph/{movies}/image/person/representations_arcface.pkl"
        else:
            model_loc = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/person/representations_arcface.pkl"

        
        if os.path.isfile(model_loc):
            os.remove(model_loc)
            #print("model deleted")
        

        #knowledge_frame = {'persons': [],'emotions':[], 'shot_name': [], 'location': [], 'places365' : [], 'timestamp': [], 'scene_name':[]}
        #knowledge_df = pd.DataFrame(data=knowledge_frame)

        # vision evaluation of dataset
        unknown_counter = 0
        for i in tqdm(range(len(scenelist))):
            num = i+1
            list_index = orderedshotlist.index(str(num))
            shots = scenelist[list_index]
            for shot in os.listdir(f"{img_path}/{movies}/{shots}"):
                border_list = []
                for image in os.listdir(f"{img_path}/{movies}/{shots}/{shot}"):
                    path = f"{img_path}/{movies}/{shots}/{shot}"
                    if testset:
                        loc_path = f"{hlvu_location}/Queries/movie_knowledge_graph/{movies}/image/location"
                        faces, emotions, unknown_counter = evaluation(image, path, movies, unknown_counter, hlvu_location, testset = testset)
                    else:
                        loc_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/location"
                        faces, emotions, unknown_counter = evaluation(image, path, movies, unknown_counter, hlvu_location, testset = testset)
                    location = compare(image, path,loc_path, delf)
                    places365_data = places365.run_places365(f"{path}/{image}", dir_path)
                    timestamp = get_timestamp_from_shot(movies, shots, image, hlvu_location)
                    vision_db.insert(
                        {'faces': faces, 'emotions': emotions, 'image': image,'location': location, 'places365':places365_data,'timestamp': timestamp, 'scene': shots, 'shots': shot})
                    torch.cuda.empty_cache()
