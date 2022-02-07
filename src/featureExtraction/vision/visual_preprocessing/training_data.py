import os
import shutil
from tqdm import tqdm
import json
from distutils.dir_util import copy_tree



def data_creation(hlvu_location):

    dir = f"{hlvu_location}/movie_knowledge_graph"
    entity_type = {}



    for movies in tqdm(os.listdir(dir)):
        if os.path.isdir(f"{dir}/{movies}"):
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
                            #print("Leerzeichen entfernt")
                        for element in lst:
                            if element != "":
                                comp_lst = element.replace(" ","").split(":")
                                entity_type[movies][comp_lst[0].replace("'","_")] = comp_lst[1]

            for file in os.listdir(f"{dir}/{movies}/image"):
                copy_tree(f"{dir}/{movies}/image",f"{dir}/{movies}/image_copy")
                # get all but the last 8 characters to remove
                # the index number and extension
                print(file)
                dir_name = file[:-6].replace(" ","").replace("'", "_")
                print(movies,dir_name)

                ent_type = entity_type[movies][dir_name]

                print(f'dir_name: {dir_name}')

                dir_path = f"{dir}/{movies}/image/{ent_type}/{dir_name}"
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


    with open('result.json', 'w') as fp:
        json.dump(entity_type, fp)

