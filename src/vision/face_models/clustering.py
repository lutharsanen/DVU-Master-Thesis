from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.cluster import DBSCAN
from tinydb import TinyDB, Query

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
        #print("return prob is: ", prob)
        if face is not None and prob>0.90: # if face detected and porbability > 90%
            emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
            #print(emb)
            df = df.append({"embedding":emb.cpu().detach().numpy()[0], "name": image},ignore_index=True)
    
    X = np.asarray(df['embedding'].values.tolist())
    
    for epsilon in np.arange(0.3, 4, 0.1):
        clustering = DBSCAN(eps=epsilon, min_samples = 3).fit(X)
        cluster = clustering.labels_
        df['dbscan'] = cluster.tolist()
        if len(set(cluster)) > 4:
            return df

def crop_unrecognized_faces(image, unknown_counter,cluster_path):
    for i in range(2):
        mtcnn = MTCNN(image_size=160, margin=10, keep_all=True, min_face_size=20)
        img = Image.open(image)       
        rgb_img = img.convert("RGB")
        boxes = mtcnn.detect(rgb_img)
        
    try:
        for box in boxes[0]:
            im1 = img.crop(list(box))
            rgba_img = im1.convert("RGB")
            rgba_img.save(f"{cluster_path}/unknown_{unknown_counter}.jpg")
            unknown_counter += 1

        return unknown_counter
    except:
        return unknown_counter


def replace_unknown_images(file, db_file):
    df = pd.read_json(file)
    X = np.asarray(df['embedding'].values.tolist())
    clustering = DBSCAN(eps=0.8, min_samples = 3).fit(X)
    cluster = clustering.labels_
    df['dbscan'] = cluster.tolist()
    labels = [i.partition("_")[0] for i in df["name"]]
    df["labels"] = labels
    test = df.groupby(["dbscan", "labels"]).size().reset_index(name="freq")
    label_class = {}
    test = test[test.dbscan != -1]
    while len(test) > 0:
        idx = test['freq'].idxmax()
        maxi = test.loc[[idx]]
        dbscan, labels, _ = maxi.to_numpy().tolist()[0]
        label_class[dbscan] = labels
        test = test[(test.dbscan != dbscan) & (test.labels != labels)]

    unknown = df.loc[df['labels'] == "unknown"]
    for row in unknown.rows:
        label_class[df.loc[df['name'] == row["name"]]["dbscan"].tolist()[0]]
        db2 = TinyDB(db_file)
        User = Query()
        db2.update({"faces":"chocolate"}, User.faces.any([row["name"]]))
        result  = db2.search(User.faces.any([row["name"]]))
        if len(result)> 0:
            lst = result[0]["faces"]
            index = result[0]["faces"].index(row["name"])
            lst[index] = "Peter"
            db2.update({"faces":lst}, User.faces.any(['unknown_2']))