import os
import numpy as np
import shlex
import subprocess
import sys
import wave
import torchaudio
import torch
import pandas as pd
from tqdm import tqdm
import string
from torch.utils.data import Subset

os.environ['CUDA_VISIBLE_DEVICES']='0'

from deepspeech import Model, version
from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote
import settings as s

model_path = "deepspeech-0.9.3-models.pbmm"
scorer_path = "deepspeech-0.9.3-models.scorer"
dataset_location = s.LIBRI_DATA_LOC
beam_width = 500
lm_alpha = 0.93
lm_beta = 1.18

model = Model(model_path)
model.enableExternalScorer(scorer_path)
model.setScorerAlphaBeta(lm_alpha, lm_beta)
model.setBeamWidth(beam_width)

def read_wav_file(audio):

    with wave.open(audio, "rb") as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)

    return buffer, rate

def transcribe(audio_file):
    buffer, rate = read_wav_file(audio_file)
    data16 = np.frombuffer(buffer, dtype=np.int16)
    return model.stt(data16)


libri_data = torchaudio.datasets.LIBRITTS(".", url = 'test-clean', folder_in_archive = 'LibriTTS', download = True)

num_train_samples = 1000
sample_ds = Subset(libri_data, np.arange(num_train_samples))

d = {'text_groundtruth': [], 'text_predicted': []}
df_text = pd.DataFrame(data=d)

for data in tqdm(sample_ds):
    file_path = f"{dataset_location}/{data[4]}/{data[5]}/{data[6]}.wav"
    text_predicted = transcribe(file_path).translate(str.maketrans('', '', string.punctuation)).lower()
    ground_truth = data[3].translate(str.maketrans('', '', string.punctuation)).lower()
    df_text.loc[df_text.shape[0]] = [ground_truth, text_predicted]

df_text.to_json("deepspeech.json")
