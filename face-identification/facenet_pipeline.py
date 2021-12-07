# importing libraries
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

mtcnn = MTCNN(image_size=240, margin=0, select_largest=False, keep_all=True, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion


def face_match(img_path, data_path): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
    img = Image.open(img_path)
    rgb_img = img.convert("RGB")
    faces, prob = mtcnn(rgb_img, return_prob=True) # returns cropped face and probability
    for face in faces:
        emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
        #print(emb)
    
        saved_data = torch.load('data.pt') # loading data.pt file
        embedding_list = saved_data[0] # getting embedding data
        name_list = saved_data[1] # getting list of names
        dist_list = [] # list of matched distances, minimum distance is used to identify the person
    
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            #print(dist)
            dist_list.append(dist)
            #print(dist_list)
        idx_min = dist_list.index(min(dist_list))
        print(name_list[idx_min], min(dist_list))

#help(MTCNN)

result = face_match('/media/lkunam/My Passport/HLVU/training/movie.keyframes/shot_keyf/honey-20/shot_0001_img_0.jpg', 'data.pt')

#print('Face matched with: ',result[0], 'With distance: ',result[1])
