import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import pandas as pd
import os
import json
from tqdm import tqdm
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from deepface import DeepFace
import numpy as np
import collections
import settings as s
import tensorflow as tf


def run():

    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

    def collate_fn(x):
        return x[0]

    test_dir = f"{s.EVAL_LOC}/bollywood_celeb_faces/testing"
    dataset=datasets.ImageFolder(test_dir) # photos folder path
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

    chosen_model = models[2]

    testloader = DataLoader(dataset, collate_fn=collate_fn)
    d = {'face_groundtruth': [], 'face_predicted': []}
    df_face = pd.DataFrame(data=d)

    for image,idx in tqdm(testloader):
        df = DeepFace.find(img_path = np.array(image), model_name = chosen_model, db_path = f"{s.EVAL_LOC}/bollywood_celeb_faces/training", detector_backend = backends[4], enforce_detection = False)
        if len(df) > 0:
            db_path = "/bollywood_celeb_faces/training/"
            index = len(s.EVAL_LOC) + len(db_path)
            name_list = list(i[index:].partition('/')[0] for i in df["identity"][:5])
            c = collections.Counter(name_list)
            name, count = c.most_common()[0]
            if count > 2:
                face_predicted = name
            else:
                face_predicted = "unknown"
            ground_truth = idx_to_class[idx]
            df_face.loc[df_face.shape[0]] = [ground_truth, face_predicted]
        
    df_face.to_json(f"deepface_{chosen_model}.json")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=2500)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    run()
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)