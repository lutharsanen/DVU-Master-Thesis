import chunk
from huggingface_hub import HfApi
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Inference, Pipeline
import pandas as pd
from pydub import AudioSegment
from spectralcluster import SpectralClusterer
import numpy as np
import models.speechdiarization.get_audiochucks as gac

available_pipelines = [p.modelId for p in HfApi().list_models(filter="pyannote-audio-pipeline")]
available_pipelines


def scene_diarization(audio_path, chunk_loc):

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    output = pipeline(audio_path)
    audio = AudioSegment.from_wav(audio_path)
    if len(output) != 0:
        for segment in output.for_json():
            speech_diarization = {'name': [], "start": [], "end": [], "label": []}
            df = pd.DataFrame(data=speech_diarization)
            for idx,segments in enumerate(output.for_json()["content"]):
                name = f"chunk_{idx}.wav"
                start = segments["segment"]["start"]
                end = segments["segment"]["end"]
                gac.generate_chunk(audio, start, end, chunk_loc, idx)
                label = segments["label"]
                df.loc[df.shape[0]] = [ name, start, end, label]
                df.to_json('speech_diarization.json')
    
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

        inference = Inference("pyannote/embedding",window="whole")
        counter = 0
        speech_embeddings = {'embedding': [], 'name': [], "start": [], "end": []}
        df = pd.DataFrame(data=speech_embeddings)
        for excerpt in vad.itertracks(yield_label=False):
            segment = excerpt[0]
            start = segment.for_json()["start"]
            end = segment.for_json()["end"]
            gac.generate_chunk(audio, start, end, chunk_loc, counter)
            try:
                embedding = inference.crop(audio_path, segment)
                #print(len(embedding))
                df.loc[df.shape[0]] = [embedding, f"chunk_{counter}.wav", start, end]
                counter += 1
            except:
                pass
        
        #check for NaN values
        for idx,value in enumerate(df["embedding"]):
            if pd.isna(value[0]):
                df = df.drop(idx)

        X = np.asarray(df["embedding"].values.tolist())
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
        df.to_json('speech_diarization.json')