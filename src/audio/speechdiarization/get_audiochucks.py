import chunk
from huggingface_hub import HfApi
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Inference, Pipeline
import pandas as pd
from pydub import AudioSegment
from spectralcluster import SpectralClusterer
import numpy as np
#from audio.speechdiarization.audio_chucks as ac
from audio.speechbrain.run import get_speech_emotion, get_speech_to_text
import shutil
import os
from os import path
from transformers import Wav2Vec2FeatureExtractor 
from datetime import datetime, timedelta
import torch

available_pipelines = [p.modelId for p in HfApi().list_models(filter="pyannote-audio-pipeline")]


def get_timestamp(movie, movie_scene, hlvu_location, segment, diarization = True):
    path = f"{hlvu_location}/scene.segmentation.reference/{movie}.csv"
    df_scenes = pd.read_csv(path, header=None)
    scene_ind = int(movie_scene.split("-")[1]) -1
    scene_start_time = datetime.strptime(df_scenes.iloc[[scene_ind]][0].to_list()[0], '%H:%M:%S')
    if diarization:
        start_delta = timedelta(seconds=segment["segment"]["start"])
        end_delta = timedelta(seconds=segment["segment"]["end"])
    else:
        start_delta = timedelta(seconds=segment["start"])
        end_delta = timedelta(seconds=segment["end"])
    starttime = scene_start_time + start_delta
    endtime = scene_start_time + end_delta
    return starttime, endtime



def scene_diarization(audio_path, chunk_loc,name, audio_db, movie, hlvu_location, code_loc):
    audio_name = name.partition(".")[0]
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    #print(audio_path)
    output = pipeline(audio_path)
    audio = AudioSegment.from_wav(audio_path)
    if len(output) != 0:

        for idx,segments in enumerate(output.for_json()["content"]):
            chunk_name = f"chunk_{idx}.wav"
            start, end = get_timestamp(movie, audio_name, hlvu_location , segments)
            #print(segments["segment"]["start"], segments["segment"]["end"])
            generate_chunk(audio, segments["segment"]["start"], segments["segment"]["end"], chunk_loc, chunk_name)
            try:
                text = get_speech_to_text(f"{chunk_loc}/{chunk_name}")
            except:
                text = "unknown"
            try:
                emotion = get_speech_emotion(f"{chunk_loc}/{chunk_name}")
            except:
                emotion = "unknown"
            label = segments["label"]
            audio_db.insert(
                {'chunk_name': chunk_name, 'start': start, 'end': end, 'label': label, 'emotion': emotion[0], 'text': text, 'scene': name})
            if path.exists(f"{code_loc}/{chunk_name}"):
                os.remove(f"{code_loc}/{chunk_name}")

            # empty not used gpu storage
            torch.cuda.empty_cache()
    
    """
    else:
        pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")
        HYPER_PARAMETERS = {
          # onset/offset activation thresholds
          "onset": 0.5, "offset": 0.5,
          # remove speech regions shorter than that many seconds.
          "min_duration_on": 0.1,
          # fill non-speech regions shorter than that many seconds.
          "min_duration_off": 0.2
        }
        pipeline.instantiate(HYPER_PARAMETERS)
        vad = pipeline(audio_path)
        if len(vad) > 2:
            inference = Inference("pyannote/embedding",window="whole")
            counter = 0
            speech_embeddings = {'embedding': [], 'name': [], "start": [], "end": [],  "emotion":[], "text":[], "scene": []}
            df = pd.DataFrame(data=speech_embeddings)
            for excerpt in vad.itertracks(yield_label=False):
                segment = excerpt[0].for_json()
                start, end = get_timestamp(movie, audio_name, hlvu_location , segment, False)
                #start_chunk = segment.for_json()["start"]
                #end_chunk = segment.for_json()["end"]
                float_start, float_end = start.timestamp(), end.timestamp()
                print(type(float_start), type(float_end))
                chunk_name = f"chunk_{counter}.wav"
                generate_chunk(audio, float_start, float_end, chunk_loc, chunk_name)
                try:
                    text = get_speech_to_text(f"{chunk_loc}/{chunk_name}")
                except:
                    text = "unknown"
                try:
                    emotion = get_speech_emotion(f"{chunk_loc}/{chunk_name}")
                except:
                    emotion = "unknown"
                os.remove(f"{code_loc}/{chunk_name}")
                try:
                    embedding = inference.crop(audio_path, segment)
                    df.loc[df.shape[0]] = [embedding, chunk_name, start, end, emotion[0], text, name]
                    counter += 1
                except:
                    pass

                # empty not used gpu storage
                torch.cuda.empty_cache()
            
            #check for NaN values
            for idx,value in enumerate(df["embedding"]):
                if pd.isna(value[0]):
                    df = df.drop(idx)

            X = np.asarray(df["embedding"].values.tolist())
            #print(len(X))
            clusterer = SpectralClusterer(
                min_clusters=2,
                max_clusters=100,
                autotune=None,
                laplacian_type=None,
                refinement_options=None,
                custom_dist="cosine")

            labels = clusterer.predict(X)

            df["label"] = labels
            df = df.drop(['embedding'], axis=1)
            dataframe_to_db(df, audio_db)
            #df.to_json(f'{data_path}/speech_diarization_{audio_name}.json')
    """

def generate_chunk( audio, start, end, chunk_loc, name):
    # convert timestamp to ms and add margin of 1s
    #print(type(start), type(end))
    start_chunk = (start *1000) -1000
    # convert timestamp to ms and add margin of 1s
    end_chunk = (end *1000) +1000
    audio_chunk=audio[start_chunk:end_chunk]
    audio_chunk.export(f"{chunk_loc}/{name}", format="wav")

def dataframe_to_db(df, audio_db):
    for _, row in df.iterrows():
        audio_db.insert(
            {'chunk_name': row["name"], 'start':row["start"], 'end': row["end"], 'label':row["label"], 'emotion': row["emotion"], "text": row["text"], "scene": row["scene"]})