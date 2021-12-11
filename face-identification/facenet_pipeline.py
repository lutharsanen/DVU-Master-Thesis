# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import os

mtcnn = MTCNN(image_size=240, margin=0, select_largest=False, keep_all=True, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion


def most_common(lst):
    return max(set(lst), key=lst.count)

def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
    for shots in os.listdir(img_path):
        persons = []
        for shot in os.listdir(f"{img_path}/{shots}"):
            img = Image.open(f"{img_path}/{shots}/{shot}")
            rgb_img = img.convert("RGB")
            faces, prob = mtcnn(rgb_img, return_prob=True) # returns cropped face and probability
            boxes, _ = mtcnn.detect(rgb_img)
            if faces != None:
                for face in faces:
                    emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
                    #print(emb)

                    saved_data = torch.load('data.pt') # loading data.pt file
                    embedding_list = saved_data[0] # getting embedding data
                    name_list = saved_data[1] # getting list of 
                    dist_list = [] # list of matched distances, minimum distance is used to identify the person

                    for idx, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        #print(dist)
                        dist_list.append(dist)
                        #print(dist_list)
                        
                    
                    idx_min = dist_list.index(min(dist_list))
                    idx_second_min = dist_list.index(sorted(dist_list)[1])
                    idx_third_min = dist_list.index(sorted(dist_list)[2])
                    persons.append(name_list[idx_min])
                    persons.append(name_list[idx_second_min])
                    persons.append(name_list[idx_third_min])
                    
                    #print(name_list[idx_min], min(dist_list))
        if len(persons) > 0: 
            print(persons)
            print(f"{shots}:{most_common(persons)}")

#help(MTCNN)

result = face_match('/media/lkunam/DVU-Challenge/HLVU/training/movie.keyframes/shot_keyf/honey-1', 'data.pt')

#print('Face matched with: ',result[0], 'With distance: ',result[1])
