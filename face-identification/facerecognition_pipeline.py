import face_recognition
import os
import json
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

#known_image = face_recognition.load_image_file("biden.jpeg")
#
##print(known_image)
#unknown_image = face_recognition.load_image_file("unknown.jpeg")
#
#biden_encoding = face_recognition.face_encodings(known_image)[0]
#
#print(biden_encoding)
#
#unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
#
#results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
#print(results)

character_information = "/media/lkunam/My Passport/HLVU/training/movie_knowledge_graphs/honey/images_copy"
movie_entity_details = "/media/lkunam/My Passport/HLVU/training/movie_knowledge_graphs/honey/honey.entity.types.txt"


f = open(movie_entity_details, "r").read()
lst = f.split("\n")
entity_type = {}
face_features = {}

# detect type of entities
for element in lst[:-1]:
    comp_lst = element.replace(" ","").split(":")
    entity_type[comp_lst[0]] = comp_lst[1]

# extract face features from chracters
for filename in tqdm(os.listdir(character_information)):
    entity_name = filename[:-6]
    if entity_type[entity_name] == "Person":
        if entity_name not in face_features.keys():
            face_features[entity_name] = []
        try:
            image = Image.open(character_information + "/" + filename)
            width, height = image.size
            new_image = image.resize((3*width, 3*height))
            new_image.save(character_information + "/" + filename)
            image_features = face_recognition.load_image_file(character_information + "/" + filename)
            image_encoding = face_recognition.face_encodings(image_features)[0].tolist()
            face_features[entity_name].append(image_encoding)
        except:
            print("No feature vectores found in", filename)
            continue

print(type(face_features))

with open("face_features.json", "w") as outfile:
    json.dump(face_features, outfile)

with open("entity_type.json", "w") as outfile:
    json.dump(entity_type, outfile)



