import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torchaudio
import torch
from speechbrain.pretrained.interfaces import foreign_class
from speechbrain.pretrained import EncoderDecoderASR
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import string
from torch.utils.data import Subset
import numpy as np
import settings as s


libri_data = torchaudio.datasets.LIBRITTS(".", url = 'test-clean', folder_in_archive = 'LibriTTS', download = True)

dataset_location = s.LIBRI_DATA_LOC

num_train_samples = 1000
sample_ds = Subset(libri_data, np.arange(num_train_samples))


def get_speech_emotion(audio_file):
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
        pymodule_file="custom_interface.py", 
        classname="CustomEncoderWav2vec2Classifier")
    _, score, _, text_lab = classifier.classify_file(audio_file)
    return text_lab, score


def get_speech_to_text(audio_file):
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="./pretrained_ASR")
    text = asr_model.transcribe_file(audio_file)
    return text


d = {'text_groundtruth': [], 'text_predicted': []}
df_text = pd.DataFrame(data=d)

for data in tqdm(sample_ds):
    file_path = f"{dataset_location}/{data[4]}/{data[5]}/{data[6]}.wav"
    text_predicted = get_speech_to_text(file_path)
    text_predicted = text_predicted.translate(str.maketrans('', '', string.punctuation)).lower()
    ground_truth = data[3].translate(str.maketrans('', '', string.punctuation)).lower()
    df_text.loc[df_text.shape[0]] = [ground_truth, text_predicted]
    os.remove(f"{data[6]}.wav")
    
df_text.to_json("speechbrain.json")
