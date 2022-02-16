import os
#os.environ['CUDA_VISIBLE_DEVICES']='4'
import audio.audio_preprocessing.extract_audio as extractor
import settings as s
from audio import scene_diarization
from tqdm import tqdm
import torch


torch.cuda.set_device(0)
# extract audio from video
#extractor.run_extractor()
hlvu_location = s.HLVU_LOCATION
audio_path = f"{hlvu_location}/audio"

audio_chunk_path = f"{hlvu_location}/audiochunk"

if not os.path.exists(audio_chunk_path):
    os.mkdir(audio_chunk_path)

movie = "honey"
custom_chunk_path = f"{hlvu_location}/audiochunk/{movie}"
if not os.path.exists(custom_chunk_path):
    os.mkdir(custom_chunk_path)
if not os.path.exists(f"{audio_path}/{movie}/data"):
    os.mkdir(f"{audio_path}/{movie}/data")


for audio_file in tqdm(os.listdir(f"{audio_path}/{movie}")[:3]):
    audio_name = audio_file.partition(".")[0]
    chunk_part_path = f"{custom_chunk_path}/{audio_name}"
    if not os.path.exists(chunk_part_path):
        os.mkdir(chunk_part_path)
    scene_diarization(f"{audio_path}/{movie}/{audio_file}", chunk_part_path, audio_file, f"{audio_path}/{movie}/data")

