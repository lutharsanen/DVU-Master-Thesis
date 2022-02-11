from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import Counter
import shutil
import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from deepface import DeepFace

# systemenv is somehow broken
import vision as crop


resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection

def create_face_db(training_dir):
    resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

    dataset=datasets.ImageFolder(training_dir) # photos folder path 
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

    def collate_fn(x):
        return x[0]

    loader = DataLoader(dataset, collate_fn=collate_fn)

    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

    for img, idx in tqdm(loader):
        face, prob = mtcnn(img, return_prob=True)
        if face is not None and prob>0.90: # if face detected and porbability > 90%
            emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
            embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
            name_list.append(idx_to_class[idx]) # names are stored in a list

    data = [embedding_list, name_list]

    torch.save(data, 'data.pt')



def detect_faces(image):
    for i in range(2):
        mtcnn = MTCNN(image_size=160, margin=10, keep_all=True, min_face_size=20)
        img = Image.open(image)       
        rgb_img = img.convert("RGB")
        boxes = mtcnn.detect(rgb_img)
    return boxes


def face_match(img):
    # getting embedding matrix of the given img
    #img = Image.open(img)
    rgb_img = img.convert("RGB")
    face, prob = mtcnn(rgb_img, return_prob=True) # returns cropped face and probability
    if face is not None and prob>0.90:
    #print(mtcnn(rgb_img, return_prob=True))
        emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false

        saved_data = torch.load('data.pt') # loading data.pt file
        embedding_list = saved_data[0] # getting embedding data
        name_list = saved_data[1] # getting list of names
        dist_list = [] # list of matched distances, minimum distance is used to identify the person

        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)

        idx_min = dist_list.index(min(dist_list))
        return (name_list[idx_min], min(dist_list))
    
    else:
        return "unknown","unknown"



def euclidean_distance_check(border, border_list):
    result_border = []
    #print(border_list)
    borders = [i[0] for i in border_list]
    for compare_border in borders:
        dist = np.linalg.norm(np.array(border) - np.array(compare_border))
        if dist < 25:
            border_list_index = borders.index(compare_border)
            result_border.append(border_list[border_list_index])
    if len(result_border) == 0:
        return 0
    #print(result_border)
    resulting_name = [i[2] for i in result_border]
    c = Counter(resulting_name)
    value, count = c.most_common()[0]
    return value


def enlargen_image(border):
    border[0]-=20
    border[1]-=20
    border[2]+=20
    border[3]+=20
    return border

def training(image_path, movies, hlvu_location,unknown_counter=None, cluster_path=None):
    border_list = []
    for image in os.listdir(image_path):
        borders, probs  = detect_faces(f"{image_path}/{image}")
        if len(borders) > 0:
            #faces = []
            for border in borders:
                im = Image.open(f"{image_path}/{image}")
                im1 = im.crop(border)
                im1.save("cropped.jpg")
                face, prob = face_match("cropped.jpg")
                if len(df) > 0:
                    path_to_db = f"/movie_knowledge_graph/{movies}/image/Person/"
                    idx = len(hlvu_location)+ len(path_to_db)
                    name_list = list(i[idx:].partition('/')[0] for i in df["identity"][:5])
                    c = Counter(name_list)
                    name, count = c.most_common()[0]
                    if count > 2:
                        #name = "{img_path}/{movies}/{shots}/{shot}/{image}"
                        shutil.copyfile("cropped.jpg", f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Person/{name}/{name}_new{image[:-4]}.png")
                        border_list.append([border,image,name])
                    else:
                        result_name = euclidean_distance_check(border, border_list)
                        if result_name!=0:
                            shutil.copyfile("cropped.jpg", f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Person/{result_name}/{result_name}_new{image[:-4]}.png")
                            border_list.append([border,image,name])


def evaluation(image, image_path, movies, unknown_counter, hlvu_location, cluster_path):
    resp = detect_faces(f"{image_path}/{image}")
    faces = []
    emotions = []
    if len(resp) > 0 and type(resp) == dict:
        for face in resp:
            border = resp[face]["facial_area"]
            enlarged_border = enlargen_image(border)
            im = Image.open(f"{image_path}/{image}")
            im1 = im.crop(enlarged_border)
            im1.save("cropped.jpg")
            obj = DeepFace.analyze(img_path = "cropped.jpg", actions = ['emotion'])
            df = DeepFace.find(img_path = "cropped.jpg", db_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Person/", detector_backend = backends[4],enforce_detection = False)
            name_list = list(i[74:].partition('/')[0] for i in df["identity"][:5])
            c = Counter(name_list)
            name, count = c.most_common()[0]
            if count > 2:
                #name = "{img_path}/{movies}/{shots}/{shot}/{image}"
                faces.append(name)
                emotions.append(obj["dominant_emotion"])
            #else:
            #    faces.append(f"unknown_{unknown_counter}.jpg")
            #    emotions.append(obj["dominant_emotion"])
    else:
        unknown_counter = crop.crop_unrecognized_faces(f"{image_path}/{image}", unknown_counter,cluster_path)
                
    return faces, emotions, unknown_counter
