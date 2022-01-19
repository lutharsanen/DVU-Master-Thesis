
from helper import match_images
import os
from tqdm import tqdm

def compare_location(image1, image_path, loc_path):
    for location in os.listdir(loc_path):
        for loc_image in os.listdir(f"{loc_path}/{location}"):
            if match_images(f"{image_path}/{image1}",f"{loc_path}/{location}/{loc_image}"):
                print(f"{image1} is in {location}.")
                return


movies = "honey"

location_path = f"/media/lkunam/DVU-Challenge/HLVU/movie_knowledge_graph/{movies}/image/Location"
img_path = "/media/lkunam/DVU-Challenge/HLVU/keyframes/shot_keyf"


shotlist = os.listdir(f"{img_path}/{movies}")
orderedshotlist = [i.partition('-')[2] for i in shotlist]
for i in tqdm(range(len(shotlist))):
    num = i+1
    list_index = orderedshotlist.index(str(num))
    shots = shotlist[list_index]
    for shot in os.listdir(f"{img_path}/{movies}/{shots}"):
        border_list = []
        for image in os.listdir(f"{img_path}/{movies}/{shots}/{shot}"):
            ###########################################################
            compare_location(image, f"{img_path}/{movies}/{shots}/{shot}", location_path)


