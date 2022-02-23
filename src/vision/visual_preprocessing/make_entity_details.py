import os
from tqdm import tqdm
import json

dir = "/media/lkunam/DVU-Challenge/HLVU/movie_knowledge_graph/Time Expired/TimeExpired.entity.types.txt"
entity_type = {}

#for movies in tqdm(os.listdir(dir)):
#        # detect type of entities
#        entity_type[movies] = {}
#        for file in os.listdir(f"{dir}/spiritualContact"):
#            if file.endswith(".entity.types.txt"):
f = open(f"{dir}", "r").read()
lst = f.split("\n")
print(lst)
for element in lst[:-1]:
   comp_lst = element.replace(" ","").split(":")
   print(comp_lst[0],comp_lst[1])
   entity_type[comp_lst[0]] = comp_lst[1]


with open('result.json', 'w') as fp:
    json.dump(entity_type, fp)

