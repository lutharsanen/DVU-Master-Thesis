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

available_pipelines = [p.modelId for p in HfApi().list_models(filter="pyannote-audio-pipeline")]


def scene_diarization(audio_path, chunk_loc,name, data_path, audio_db):
    #print(audio_path)
    audio_name = name.partition(".")[0]
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    output = pipeline(audio_path)
    audio = AudioSegment.from_wav(audio_path)
    if len(output) != 0:
        for segment in output.for_json():
            speech_diarization = {'name': [], "start": [], "end": [], "label": [], "emotion":[], "text":[]}
            #speech_diarization = {'name': [], "start": [], "end": [], "label": []}
            df = pd.DataFrame(data=speech_diarization)
            for idx,segments in enumerate(output.for_json()["content"]):
                name = f"chunk_{idx}.wav"
                start = segments["segment"]["start"] 
                end = segments["segment"]["end"]
                generate_chunk(audio, start, end, chunk_loc, name)
                text = get_speech_to_text(f"{chunk_loc}/{name}")
                emotion = get_speech_emotion(f"{chunk_loc}/{name}")
                label = segments["label"]
                df.loc[df.shape[0]] = [ name, start, end, label, emotion, text]
                #df.loc[df.shape[0]] = [ name, start, end, label]
        df.to_csv(f'{data_path}/speech_diarization_{audio_name}.csv')
    
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
        if len(vad) > 0:
            inference = Inference("pyannote/embedding",window="whole")
            counter = 0
            speech_embeddings = {'embedding': [], 'name': [], "start": [], "end": [],  "emotion":[], "text":[]}
            #speech_embeddings = {'embedding': [], 'name': [], "start": [], "end": []}
            df = pd.DataFrame(data=speech_embeddings)
            for excerpt in vad.itertracks(yield_label=False):
                segment = excerpt[0]
                start = segment.for_json()["start"]
                end = segment.for_json()["end"]
                name = f"chunk_{counter}.wav"
                generate_chunk(audio, start, end, chunk_loc, name)
                #print(f"{chunk_loc}/{name}")
                text = get_speech_to_text(f"{chunk_loc}/{name}")
                emotion = get_speech_emotion(f"{chunk_loc}/{name}")
                #print(text,emotion)
                os.remove(name)
                try:
                    embedding = inference.crop(audio_path, segment)
                    #print(len(embedding))
                    df.loc[df.shape[0]] = [embedding, name, start, end, emotion, text]
                    #df.loc[df.shape[0]] = [embedding, name, start, end]
                    counter += 1
                except:
                    pass
            
            #check for NaN values
            for idx,value in enumerate(df["embedding"]):
                if pd.isna(value[0]):
                    df = df.drop(idx)

            X = np.asarray(df["embedding"].values.tolist())
            print(len(X))
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
            df.to_json(f'{data_path}/speech_diarization_{audio_name}.json')

def generate_chunk( audio, start, end, chunk_loc, name):
    # convert timestamp to ms and add margin of 1s
    start_chunk = (start *1000) -1000
    # convert timestamp to ms and add margin of 1s
    end_chunk = (end *1000) +1000
    audio_chunk=audio[start_chunk:end_chunk]
    audio_chunk.export(f"{chunk_loc}/{name}", format="wav")
