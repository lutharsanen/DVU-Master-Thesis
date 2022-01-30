from tqdm import tqdm
import os
from models.deepface import deepface_logic as deepface
from models.DELF import run_delf as delf
from visual_preprocessing import training_data as training
from models.facenet import clustering
import settings as s
import pandas as pd

hlvu_location = s.HLVU_LOCATION
img_path = f"{hlvu_location}/keyframes/shot_keyf"

#training.data_creation()


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
            border_list = deepface.training(image, path, movies, border_list, hlvu_location)

# delete existing model to generate new one
model_loc = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Person/representations_vgg_face.pkl"

if os.path.isfile(model_loc):
    os.remove(model_loc)
    #print("model deleted")


cluster_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/clustering"
os.mkdir(cluster_path)

knowledge_frame = {'persons': [], 'shot_name': [], 'location': []}
knowledge_df = pd.DataFrame(data=knowledge_frame)

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
            loc_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Location/"
            faces = deepface.evaluation(image, path, movies, unknown_counter, hlvu_location, cluster_path)
            location = delf.compare(image, path,loc_path)
            knowledge_df = knowledge_df.append({"persons":faces, "name": image, "location": location},ignore_index=True)

cluster_df = clustering.start_face_clustering(cluster_path)
# then cluster with facenet and replace unknown images by person images

knowledge_df.to_json(f"knowledge_{movies}.json")
cluster_df.to_json(f"cluster_{movies}.json")