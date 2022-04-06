import os
os.environ['CUDA_VISIBLE_DEVICES']='5'
import audio.audio_preprocessing.extract_audio as extractor
import settings as s
from audio import scene_diarization
from tqdm import tqdm
import torch
from tinydb import TinyDB
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer
from tinydb.storages import JSONStorage


torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(0.4, 0)
# extract audio from video
#extractor.run_extractor()
#hlvu_location = s.HLVU_LOCATION
#audio_path = f"{hlvu_location}/audio"
#code_loc = s.DIR_PATH


#audio_chunk_path = f"{hlvu_location}/audiochunk"

#if not os.path.exists(audio_chunk_path):
#    os.mkdir(audio_chunk_path)

#movie = "Huckleberry_Finn"
#movie_list = ["Nuclear_Family"]

def audio_stream(hlvu_location, movie_list, audio_path, code_loc):
    for movie in movie_list:
        custom_chunk_path = f"{hlvu_location}/audiochunk/{movie}"

        serialization = SerializationMiddleware(JSONStorage)
        serialization.register_serializer(DateTimeSerializer(), 'TinyDate')
        audio_db = TinyDB(f'{code_loc}/database/audio_{movie}.json', storage=serialization)



        if not os.path.exists(custom_chunk_path):
            os.mkdir(custom_chunk_path)

        for audio_file in tqdm(os.listdir(f"{audio_path}/{movie}")):
            audio_name = audio_file.partition(".")[0]
            chunk_part_path = f"{custom_chunk_path}/{audio_name}"
            if not os.path.exists(chunk_part_path):
                os.mkdir(chunk_part_path)
            scene_diarization(f"{audio_path}/{movie}/{audio_file}", chunk_part_path, audio_file, audio_db, movie, hlvu_location, code_loc)
            torch.cuda.empty_cache()


