from os import name
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.cluster import DBSCAN


def start_face_clustering(clustering_path):
    #Image.fromarray(frame)
    
    mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
    resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion
    
    face_embeddings = {'embedding': [], 'name': []}
    df = pd.DataFrame(data=face_embeddings)
    
    for image in os.listdir(clustering_path):
        img = Image.open(f"{clustering_path}/{image}")
        rgb_img = img.convert("RGB")
        face, prob = mtcnn(rgb_img, return_prob=True) 
        print("return prob is: ", prob)
        if face is not None and prob>0.90: # if face detected and porbability > 90%
            emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
            print(emb)
            df = df.append({"embedding":emb.cpu().detach().numpy()[0], "name": image},ignore_index=True)
    
    X = np.asarray(df['embedding'].values.tolist())
    
    clustering = DBSCAN(eps=1, min_samples = 3).fit(X)
    cluster = clustering.labels_
    df['dbscan'] = cluster.tolist()
    return df