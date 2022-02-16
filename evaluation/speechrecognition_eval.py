import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torchaudio
import torch
import speech_recognition as sr
import pandas as pd
import settings as s
from torch.utils.data import Subset
import numpy as np
import string
from tqdm import tqdm

libri_data = torchaudio.datasets.LIBRITTS(".", url = 'test-clean', folder_in_archive = 'LibriTTS', download = False)

dataset_location = s.LIBRI_DATA_LOC
num_train_samples = 1000
sample_ds = Subset(libri_data, np.arange(num_train_samples))


def get_speech_to_text(audio):
    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Sphinx
    try:
        text = r.recognize_sphinx(audio)
    except sr.UnknownValueError:
        text = "Not Understood"
    except sr.RequestError as e:
        text = "Not Understood"
        
    return text


d = {'text_groundtruth': [], 'text_predicted': []}
df_text = pd.DataFrame(data=d)

for data in tqdm(sample_ds):
    file_path = f"{dataset_location}/{data[4]}/{data[5]}/{data[6]}.wav"
    text_predicted = get_speech_to_text(file_path)
    text_predicted = text_predicted.translate(str.maketrans('', '', string.punctuation)).lower()
    ground_truth = data[3].translate(str.maketrans('', '', string.punctuation)).lower()
    df_text.loc[df_text.shape[0]] = [ground_truth, text_predicted]
    
df_text.to_json("speech_recognition.json")
