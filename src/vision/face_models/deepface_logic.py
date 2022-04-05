from retinaface import RetinaFace
from deepface import DeepFace
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import Counter
import shutil
import os

# systemenv is somehow broken
import vision as crop


backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]


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

def training(image_path, movies, hlvu_location,unknown_counter=None, cluster_path=None, testset = False):
    border_list = []
    if testset:
        hlvu_location = f"{hlvu_location}/Queries"
    for image in os.listdir(image_path):
        resp = RetinaFace.detect_faces(f"{image_path}/{image}")
        if len(resp) > 0 and type(resp) == dict:
            #faces = []
            for face in resp:
                border = resp[face]["facial_area"]
                enlarged_border = enlargen_image(border)
                im = Image.open(f"{image_path}/{image}")
                im1 = im.crop(enlarged_border)
                im1.save("cropped.jpg")
                df = DeepFace.find(img_path = "cropped.jpg", model_name = models[6], db_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Person/", detector_backend = backends[4], enforce_detection = False)
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


def evaluation(image, image_path, movies, unknown_counter, hlvu_location, cluster_path, testset = False):
    resp = RetinaFace.detect_faces(f"{image_path}/{image}")
    faces = []
    emotions = []
    if testset:
        hlvu_location = f"{hlvu_location}/Queries"
    if len(resp) > 0 and type(resp) == dict:
        for face in resp:
            border = resp[face]["facial_area"]
            enlarged_border = enlargen_image(border)
            im = Image.open(f"{image_path}/{image}")
            im1 = im.crop(enlarged_border)
            im1.save("cropped.jpg")
            obj = DeepFace.analyze(img_path = "cropped.jpg", actions = ['emotion'], enforce_detection = False)
            #print(f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Person/")
            df = DeepFace.find(img_path = "cropped.jpg", model_name = models[6], db_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/Person/", detector_backend = backends[4], enforce_detection = False)
            path_to_db = f"/movie_knowledge_graph/{movies}/image/Person/"
            idx = len(hlvu_location)+ len(path_to_db)
            name_list = list(i[idx:].partition('/')[0] for i in df["identity"][:5])
            if len(name_list) > 0:
                c = Counter(name_list)
                name, count = c.most_common()[0]
                if count > 2:
                    #name = "{img_path}/{movies}/{shots}/{shot}/{image}"
                    faces.append(name)
                    emotions.append(obj["dominant_emotion"])
                else:
                    faces.append(f"unknown_{unknown_counter}.jpg")
                    emotions.append(obj["dominant_emotion"])
                    shutil.copyfile("cropped.jpg", f"{cluster_path}/unknown_{unknown_counter}.jpg")

    else:
        #print(f"{image_path}/{image}", unknown_counter,cluster_path)
        unknown_counter, face_detected = crop.crop_unrecognized_faces(f"{image_path}/{image}", unknown_counter,cluster_path)
        if face_detected:
            faces.append(f"unknown_{unknown_counter}.jpg")
            unknown_counter += 1
                
    return faces, emotions, unknown_counter

