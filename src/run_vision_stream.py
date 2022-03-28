import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

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

hlvu_location = s.HLVU_LOCATION
dir_path = s.DIR_PATH
img_path = f"{hlvu_location}/keyframes/shot_keyf"

def get_timestamp_from_shot(movie, movie_scene, shot_name):
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


def run(testset = False):

    #training.data_creation()

    
    #movies = "honey"
    movie_list = ["shooters", "The_Big_Something", "time_expired", "Valkaama", "Huckleberry_Finn", "spiritual_contact", "honey", "sophie", "Nuclear_Family", "SuperHero"]
    # preload delf model
    delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

    #for movies in os.listdir(img_path):
    for movies in movie_list:
        serialization = SerializationMiddleware(JSONStorage)
        serialization.register_serializer(DateTimeSerializer(), 'TinyDate')
        vision_db = TinyDB(f'database/vision_{movies}.json', storage=serialization)
        scenelist = os.listdir(f"{img_path}/{movies}")
        orderedshotlist = [i.partition('-')[2] for i in scenelist]


        # extend cluster path with original training images
        if testset:
            cluster_path = f"{hlvu_location}/Queries/movie_knowledge_graph/{movies}/clustering"
            training_image_path = f"{hlvu_location}/Queries/movie_knowledge_graph/{movies}/image/Person"
        else:
            cluster_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/clustering"
            training_image_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Person"
        if not os.path.exists(cluster_path):
            os.mkdir(cluster_path)
        for folder in os.listdir(training_image_path):
            for images in os.listdir(f"{training_image_path}/{folder}/"):
                image = f"{training_image_path}/{folder}/{images}"
                shutil.copyfile(image, f"{cluster_path}/{images}")

        # finetuning face recognition model
        for i in tqdm(range(len(scenelist))):
            num = i+1
            list_index = orderedshotlist.index(str(num))
            shots = scenelist[list_index]
            for shot in os.listdir(f"{img_path}/{movies}/{shots}"):
                path = f"{img_path}/{movies}/{shots}/{shot}"
                training(path, movies, hlvu_location)

        # delete existing model to generate new one
        if testset:
            model_loc = f"{hlvu_location}/Queries/movie_knowledge_graph/{movies}/image/Person/representations_arcface.pkl"
        else:
            model_loc = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Person/representations_arcface.pkl"

        
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
                        loc_path = f"{hlvu_location}/Queries/movie_knowledge_graph/{movies}/image/Location"
                    else:
                        loc_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Location"
                    faces, emotions, unknown_counter = evaluation(image, path, movies, unknown_counter, hlvu_location, cluster_path)
                    location = compare(image, path,loc_path, delf)
                    places365_data = places365.run_places365(f"{path}/{image}", dir_path)
                    timestamp = get_timestamp_from_shot(movies, shots, image)
                    #knowledge_df.loc[knowledge_df.shape[0]] = [faces, emotions, image, location, places365_data, timestamp, shots]
                    vision_db.insert(
                        {'faces': faces, 'emotions': emotions, 'image': image,'location': location, 'places365':places365_data,'timestamp': timestamp, 'scene': shots, 'shots': shot})
                    torch.cuda.empty_cache()

        #cluster_df = start_face_clustering(cluster_path)
        # then cluster with facenet and replace unknown images by person images

        #knowledge_df.to_json(f"knowledge_{movies}.json")
        #cluster_df.to_json(f"cluster_{movies}.json")



gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4500)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    torch.cuda.set_device(0)
    run()
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

#run()
