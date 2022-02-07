import audio.audio_preprocessing.extract_audio as extractor
import settings as s
from audio import scene_diarization
import os

# extract audio from video
#extractor.run_extractor()
hlvu_location = s.HLVU_LOCATION
audio_path = f"{hlvu_location}/audio"

audio_chunk_path = f"{hlvu_location}/audiochunk"

if not os.path.exists(audio_chunk_path):
    os.mkdir(audio_chunk_path)

movie = "honey"


for audio_file in os.listdir(f"{audio_path}/{movie}"):
    custom_chunk_path = f"{hlvu_location}/audiochunk/{movie}"
    if not os.path.exists(custom_chunk_path):
        os.mkdir(custom_chunk_path)
    scene_diarization(f"{audio_path}/{movie}/{audio_file}", custom_chunk_path)

