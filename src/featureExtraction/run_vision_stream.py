import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

from sklearn import cluster
from tqdm import tqdm
import os
from vision import evaluation, training
from vision import compare
from vision import data_creation
from vision import start_face_clustering
import settings as s
import pandas as pd
import shutil
import vision.places365.run_placesCNN_basic as places365 
import tensorflow as tf


def run():

    hlvu_location = s.HLVU_LOCATION
    dir_path = s.DIR_PATH
    img_path = f"{hlvu_location}/keyframes/shot_keyf"

    #training.data_creation()


    movies = "honey"
    shotlist = os.listdir(f"{img_path}/{movies}")
    orderedshotlist = [i.partition('-')[2] for i in shotlist]

    # finetuning face recognition model

    for i in tqdm(range(len(shotlist[1:3]))):
        num = i+1
        list_index = orderedshotlist.index(str(num))
        shots = shotlist[list_index]
        for shot in os.listdir(f"{img_path}/{movies}/{shots}"):
            path = f"{img_path}/{movies}/{shots}/{shot}"
            training(path, movies, hlvu_location)

    # delete existing model to generate new one
    model_loc = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Person/representations_vgg_face.pkl"
    
    if os.path.isfile(model_loc):
        os.remove(model_loc)
        #print("model deleted")
    

    cluster_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/clustering"
    training_image_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Person"
    if not os.path.exists(cluster_path):
        os.mkdir(cluster_path)
    for folder in os.listdir(training_image_path):
        for images in os.listdir(f"{training_image_path}/{folder}/"):
            image = f"{training_image_path}/{folder}/{images}"
            shutil.copyfile(image, f"{cluster_path}/{images}")
    

    knowledge_frame = {'persons': [],'emotions':[], 'shot_name': [], 'location': [], 'places365' : []}
    knowledge_df = pd.DataFrame(data=knowledge_frame)
#
    ## vision evaluation of dataset
    unknown_counter = 0
    for i in tqdm(range(len(shotlist[1:3]))):
        num = i+1
        list_index = orderedshotlist.index(str(num))
        shots = shotlist[list_index]
        for shot in os.listdir(f"{img_path}/{movies}/{shots}"):
            border_list = []
            for image in os.listdir(f"{img_path}/{movies}/{shots}/{shot}"):
                path = f"{img_path}/{movies}/{shots}/{shot}"
                loc_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Location"
                faces, emotions, unknown_counter = evaluation(image, path, movies, unknown_counter, hlvu_location, cluster_path)
                location = compare(image, path,loc_path)
                places365_data = places365.run_places365(f"{path}/{image}", dir_path)
                knowledge_df.loc[knowledge_df.shape[0]] = [faces, emotions, image, location, places365_data]
                

    cluster_df = start_face_clustering(cluster_path)
    # then cluster with facenet and replace unknown images by person images

    knowledge_df.to_json(f"knowledge_{movies}.json")
    cluster_df.to_json(f"cluster_{movies}.json")



gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=2500)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    run()
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

#run()
