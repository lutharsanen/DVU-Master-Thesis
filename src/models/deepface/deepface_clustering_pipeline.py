import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import os
from deepface import DeepFace
from retinaface import RetinaFace
from PIL import Image
from sklearn.cluster import DBSCAN

def enlargen_image(border):
    border[0]-=20
    border[1]-=20
    border[2]+=20
    border[3]+=20
    return border

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "Dlib", "ArcFace"]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

face_embeddings = {'embedding': [], 'name': []}
df = pd.DataFrame(data=face_embeddings)

for image in os.listdir("clustering_images"):
    resp = RetinaFace.detect_faces(f"clustering_images/{image}")
    for face in resp:
        if len(face) < 1:
            print(f"no face in {image}")
        else:
            border = resp[face]["facial_area"]
            enlarged_border = enlargen_image(border)
            im = Image.open(f"clustering_images/{image}")
            im1 = im.crop(enlarged_border)
            im1.save("cropped.jpg")
            try:
                embedding = DeepFace.represent("cropped.jpg", model_name = models[3], detector_backend = backends[4])
                embedding = np.asarray(embedding)
                print(type(embedding))
                df = df.append({"embedding":embedding, "name": image},ignore_index=True)
            except ValueError:
                print(f"face in {image} could not be detected") 


df.to_json('embeddings.json')
X = np.asarray(df['embedding'].values.tolist())
print("Start Clustering ....")
clustering = DBSCAN(eps = 1, min_samples = 5).fit(X)
cluster = clustering.labels_

print(len(set(cluster)))