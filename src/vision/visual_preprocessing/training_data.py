import os
import shutil
from tqdm import tqdm
import json
from distutils.dir_util import copy_tree



def data_creation(hlvu_location, movie_list, testset = False):
    if testset:
        hlvu_location = f"{hlvu_location}/Queries"
    dir = f"{hlvu_location}/movie_knowledge_graph"
    entity_type = {}

    for movies in tqdm(movie_list):
        if os.path.isdir(f"{dir}/{movies}"):
            print("directory found")
            print(movies)
            # detect type of entities
            if os.path.isdir(f"{dir}/{movies}"):
            # detect type of entities
                entity_type[movies] = {}
                for file in os.listdir(f"{dir}/{movies}"):
                    if file.endswith(".entity.types.txt"):
                        f = open(f"{dir}/{movies}/{file}", "r").read()
                        lst = f.split("\n")
                        if "" in lst:
                            lst.remove("")
                        for element in lst:
                            if element != "":
                                comp_lst = element.replace(" ","").split(":")
                                entity_type[movies][comp_lst[0].replace("'","_").lower()] = comp_lst[1].lower()
            print(entity_type)
            copy_tree(f"{dir}/{movies}/image",f"{dir}/{movies}/image_copy")
            print(f"{dir}/{movies}/image")
            for file in os.listdir(f"{dir}/{movies}/image"):
                # get all but the last 8 characters to remove
                # the index number and extension
                print(file)
                dir_name = file[:-6].replace(" ","").replace("'", "_").lower()
                print(dir_name)
                ent_type = entity_type[movies][dir_name]

                dir_path = f"{dir}/{movies}/image/{ent_type}/{dir_name}"
                # check if directory exists or not yet
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                if os.path.exists(dir_path):
                    file_path = f"{dir}/{movies}/image/{file}"

                    # move files into created directory
                    shutil.move(file_path, dir_path)

