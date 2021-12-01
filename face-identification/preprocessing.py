import os
import shutil

dir = "/media/lkunam/My Passport/HLVU/training/movie_knowledge_graphs/honey/images_copy"
movie_entity_details = "/media/lkunam/My Passport/HLVU/training/movie_knowledge_graphs/honey/honey.entity.types.txt"

f = open(movie_entity_details, "r").read()
lst = f.split("\n")
entity_type = {}
face_features = {}

# detect type of entities
for element in lst[:-1]:
    comp_lst = element.replace(" ","").split(":")
    entity_type[comp_lst[0]] = comp_lst[1]

for file in os.listdir(dir):
    # get all but the last 8 characters to remove
    # the index number and extension
    dir_name = file[:-6]
    print(f'dir_name: {dir_name}')

    dir_path = f"{dir}/{dir_name}"
    print(f'dir_path: {dir_path}')
    
    # check if directory exists or not yet
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists(dir_path):
        print(file)
        file_path = f"{dir}/{file}"
        print(f'file_path: {file_path}')
        
        # move files into created directory
        shutil.move(file_path, dir_path)