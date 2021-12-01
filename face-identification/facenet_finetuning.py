from os import name
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image


dir = "/media/lkunam/My Passport/HLVU/training/movie_knowledge_graphs/honey/images_copy/face"
movie_entity_details = "/media/lkunam/My Passport/HLVU/training/movie_knowledge_graphs/honey/honey.entity.types.txt"

f = open(movie_entity_details, "r").read()
lst = f.split("\n")
entity_type = {}
face_features = {}

# detect type of entities
for element in lst[:-1]:
    comp_lst = element.replace(" ","").split(":")
    entity_type[comp_lst[0]] = comp_lst[1]

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

dataset=datasets.ImageFolder(dir) # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

face_list = [] # list of cropped faces from photos folder
name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    print(img)
    face, prob = mtcnn(img, return_prob=True) 
    print("return prob is: ", prob)
    if face is not None and prob>0.90: # if face detected and porbability > 90%
        emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
        embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
        name_list.append(idx_to_class[idx]) # names are stored in a list
        
data = [embedding_list, name_list]

torch.save(data, 'data.pt')