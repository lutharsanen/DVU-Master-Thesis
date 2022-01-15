import os
from retinaface import RetinaFace
from deepface import DeepFace
from PIL import Image
import shutil


def enlargen_image(border):
    border[0]-=20
    border[1]-=20
    border[2]+=20
    border[3]+=20
    return border


img_path = "/media/lkunam/DVU-Challenge/HLVU/keyframes/shot_keyf"

#for movies in os.listdir(img_path):
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']


movies = "honey"
shotlist = os.listdir(f"{img_path}/{movies}")
orderedshotlist = [i.partition('-')[2] for i in shotlist]
for i in range(len(shotlist)):
    num = i+1
    list_index = orderedshotlist.index(str(num))
    shots = shotlist[list_index]
    for shot in os.listdir(f"{img_path}/{movies}/{shots}"):
        for image in os.listdir(f"{img_path}/{movies}/{shots}/{shot}"):
            #print(image)
            #print(f"{img_path}/{movies}/{shots}/{shot}/{image}")
            resp = RetinaFace.detect_faces(f"{img_path}/{movies}/{shots}/{shot}/{image}")
            #print(len(resp))
            #print(resp)
            if len(resp) > 0 and type(resp) == dict:
                for face in resp:
                    print(face)
                    border = resp[face]["facial_area"]
                    enlarged_border = enlargen_image(border)
                    im = Image.open(f"{img_path}/{movies}/{shots}/{shot}/{image}")
                    im1 = im.crop(border)
                    im1.save("cropped.jpg")
                    try:
                        df = DeepFace.find(img_path = "cropped.jpg", db_path = f"/media/lkunam/DVU-Challenge/HLVU/movie_knowledge_graph/{movies}/image/Person/", detector_backend = backends[4])
                        result = df["identity"][0]
                        name = result[-17:].partition('/')[0]
                        #name = "{img_path}/{movies}/{shots}/{shot}/{image}"
                        shutil.copyfile("cropped.jpg", f"/media/lkunam/DVU-Challenge/HLVU/movie_knowledge_graph/{movies}/image/Person/{name}/{name}_new_{image[:-4]}")
                    except ValueError:
                        print("No face was able to be matched")
                        print(f"{img_path}/{movies}/{shots}/{shot}/{image}")
                        break
    