import speechbrain
import os
from speechbrain.pretrained.interfaces import foreign_class
import pandas as pd
import settings as s


def get_speech_emotion(audio_file):
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
        pymodule_file="custom_interface.py", 
        classname="CustomEncoderWav2vec2Classifier")
    _, score, _, text_lab = classifier.classify_file(audio_file)
    return text_lab, score

emotion_dir = f"{s.EVAL_LOC}/speech_emotion"

d = {'emotion_groundtruth': [], 'emotion_predicted': []}
df_speech_emotion = pd.DataFrame(data=d)

dataset = os.listdir(emotion_dir)[:1000]

for audio in dataset:
    ground_truth = audio[9:12]
    emotion_predicted, _ = get_speech_emotion(f"{emotion_dir}/{audio}")
    df_speech_emotion.loc[df_speech_emotion.shape[0]] = [ground_truth, emotion_predicted]
    os.remove(audio)
    

df_speech_emotion.to_json(f"speech_emotion.json")