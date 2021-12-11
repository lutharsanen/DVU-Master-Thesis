import os
from tqdm import tqdm

dir = "/media/lkunam/DVU-Challenge/HLVU/movie_knowledge_graph"
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
