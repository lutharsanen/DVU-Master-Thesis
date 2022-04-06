from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage # added. missing in readme.
from tinydb import TinyDB, Query
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer
import os
from datetime import datetime
from tqdm import tqdm
#from settings import *


#movie_list = ["shooters", 
#              "The_Big_Something", 
#              "time_expired", 
#              "Valkaama", 
#              "Huckleberry_Finn", 
#              "spiritual_contact", 
#              "honey", 
#              "sophie", 
#              "Nuclear_Family", 
#              "SuperHero"
#             ]

#movie_list = ["Nuclear_Family"]

def most_frequent(List):
    return max(set(List), key = List.count)

def audio_vision_combiner(movie_list,HLVU_LOCATION, DIR_PATH):
    for movies in movie_list:
        serialization1 = SerializationMiddleware(JSONStorage)
        serialization1.register_serializer(DateTimeSerializer(), 'TinyDate')
        serialization2 = SerializationMiddleware(JSONStorage)
        serialization2.register_serializer(DateTimeSerializer(), 'TinyDate')

        db_audio = TinyDB(f'{DIR_PATH}/database/audio_{movies}.json', storage=serialization1)
        db_vision = TinyDB(f'{DIR_PATH}/database/vision_{movies}.json', storage=serialization2)

        speaker_dict = {}

        audio_list = [i for i in sorted(os.listdir(f"{HLVU_LOCATION}/audio/{movies}")) if movies in i]
        for scene in tqdm(audio_list):
            speaker_dict[scene] = {}
            if movies in scene:
                query = Query()
                results = db_audio.search(query.scene == scene)
                speakers = [i["label"] for i in results]
                speaker_set = set(speakers)
                for speaker in speaker_set:
                    query = Query()

                    results = db_audio.search(query.label == speaker)

                    face_list = []

                    for result in results:
                        start = result["start"]
                        end = result["end"]
                        try:
                            answer = db_vision.search((query.timestamp > start) & (query.timestamp < end))
                            if len(answer) > 0:
                                if len(answer[0]["faces"]) > 0:
                                    for faces in answer[0]["faces"]:
                                        if not "unknown" in faces:
                                            face_list.append(faces)
                        except:
                            pass
                    if len(face_list) > 0:
                        speaker_dict[scene][speaker] = most_frequent(face_list)

        audio_db = db_audio.all()
        User = Query()
        for i in tqdm(audio_db):
            scene = i["scene"]
            speaker = i["label"]
            chunk_name = i["chunk_name"]
            name = speaker_dict[scene][speaker]
            db_audio.update({'label': name}, (User.scene == scene) & (User.chunk_name == chunk_name))
        