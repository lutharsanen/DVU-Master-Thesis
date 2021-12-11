import os
import shutil
from tqdm import tqdm

dir = "/media/lkunam/DVU-Challenge/HLVU/movie_knowledge_graph"
movie_entity_details = "/media/lkunam/DVU-Challenge/HLVU/movie_knowledge_graph/honey/honey.entity.types.txt"

f = open(movie_entity_details, "r").read()
lst = f.split("\n")
entity_type = {}



for movies in tqdm(os.listdir(dir)):
    if os.path.isdir(f"{dir}/{movies}"):
        # detect type of entities
        entity_type[movies] = {}
        f = open(f"{dir}/{movies}/{movies}.entity.types.txt", "r").read()
        lst = f.split("\n")
        for element in lst[:-1]:
           comp_lst = element.replace(" ","").split(":")
           entity_type[movies][comp_lst[0]] = comp_lst[1]

        for file in os.listdir(f"{dir}/{movies}/image"):
           # get all but the last 8 characters to remove
           # the index number and extension
           dir_name = file[:-6]

           ent_type = entity_type[dir_name]

           print(f'dir_name: {dir_name}')

           dir_path = f"{dir}/{movies}/movie/{ent_type}/{dir_name}"
           print(f'dir_path: {dir_path}')

           # check if directory exists or not yet
           if not os.path.exists(dir_path):
               os.makedirs(dir_path)

           if os.path.exists(dir_path):
               print(file)
               file_path = f"{dir}/{movies}/image/{file}"
               print(f'file_path: {file_path}')

               # move files into created directory
               shutil.move(file_path, dir_path)


