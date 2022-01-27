from tqdm import tqdm
import os
from models.deepface import deepface_logic as deepface
from models.DELF import run_delf as delf
from preprocessing import training_data as training

img_path = "/media/lkunam/DVU-Challenge/HLVU/keyframes/shot_keyf"

training.data_creation()


movies = "honey"
shotlist = os.listdir(f"{img_path}/{movies}")
orderedshotlist = [i.partition('-')[2] for i in shotlist]

# finetuning face recognition model
for i in tqdm(range(len(shotlist))):
    num = i+1
    list_index = orderedshotlist.index(str(num))
    shots = shotlist[list_index]
    for shot in os.listdir(f"{img_path}/{movies}/{shots}"):
        border_list = []
        for image in os.listdir(f"{img_path}/{movies}/{shots}/{shot}"):
            path = f"{img_path}/{movies}/{shots}/{shot}"
            border_list = deepface.training(image, path, movies, border_list)

# vision evaluation of dataset
unknown_counter = 0
for i in tqdm(range(len(shotlist))):
    num = i+1
    list_index = orderedshotlist.index(str(num))
    shots = shotlist[list_index]
    for shot in os.listdir(f"{img_path}/{movies}/{shots}"):
        border_list = []
        for image in os.listdir(f"{img_path}/{movies}/{shots}/{shot}"):
            path = f"{img_path}/{movies}/{shots}/{shot}"
            loc_path = f"/media/lkunam/DVU-Challenge/HLVU/movie_knowledge_graph/{movies}/image/Location/"
            faces = deepface.evaluation(image, path, movies, unknown_counter)
            location = delf.compare(image, path,loc_path)

# then cluster with facenet and replace unknown images by person images
