import pandas as pd
import os
import numpy as np
from tinydb import TinyDB,Query,where
from tinydb.storages import JSONStorage
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer
from tqdm import tqdm
from relationship_helper.scene_relation import get_scene_features,action_query,audio_query,vision_query,prep_data_creator
from interaction_transformer import kinetics400_to_interaction as kinetics400



def scene_data_creation(movie_list, dir_path, hlvu_location):
    d = {
        'person1': [], 
        'person2': [],
        'action': [], 
        'face_happy':[], 
        'face_angry':[], 
        'face_neutral':[],
        'face_sad':[],
        'face_surprise':[],
        'text_happy':[],
        'text_angry':[],
        'text_neutral':[],
        'text_sad':[],
        'interaction':[]
        }
    df_interaction = pd.DataFrame(data=d)


    for movie in movie_list:

        serialization_vision = SerializationMiddleware(JSONStorage)
        serialization_vision.register_serializer(DateTimeSerializer(), 'TinyDate')

        serialization_audio = SerializationMiddleware(JSONStorage)
        serialization_audio.register_serializer(DateTimeSerializer(), 'TinyDate')

        serialization_video = SerializationMiddleware(JSONStorage)
        serialization_video.register_serializer(DateTimeSerializer(), 'TinyDate')


        vision_db = TinyDB(f"{dir_path}/database/vision_{movie}.json", storage=serialization_vision)
        video_db = TinyDB(f"{dir_path}/database/action.json", storage=serialization_video)
        audio_db = TinyDB(f"{dir_path}/database/audio_{movie}.json", storage=serialization_audio)


        path = f"{hlvu_location}/keyframes/shot_keyf/{movie}"
        json_path = f"{hlvu_location}/scenes_knowledge_graphs"

        movie = os.listdir(f"{hlvu_location}/keyframes/shot_keyf/{movie}")
        for scene in tqdm(movie):
            prep_interaction, prep_emotion, prep_location = prep_data_creator(f"{json_path}/{scene}.json")
            #print(prep_interaction)
            full_list = sorted(os.listdir(f"{path}/{scene}"))
            split_nr = len(prep_interaction)
            if len(prep_interaction) > 1:
                split_list = list(np.array_split(full_list, split_nr))
            else:
                split_list = full_list
            for idx,row in prep_interaction.iterrows():
                chunk_list = split_list[idx]
                #print(row["action"],row["person1"],row["person2"],row["sequence"])
                transformed_action, emo, text_emo, text = get_scene_features(
                    row["person1"], row["person2"], scene, chunk_list, kinetics400,
                    vision_db, video_db, audio_db
                )

                interaction = row["action"]
                for action, em, t_em, tex in zip(transformed_action, emo, text_emo, text):

                    #if transformed_action != "unknown":
                    df_interaction.loc[df_interaction.shape[0]] = [
                        row["person1"], 
                        row["person2"],
                        action, 
                        em["happy"], 
                        em["angry"], 
                        em["neutral"], 
                        em["sad"], 
                        em["surprise"],
                        t_em["hap"], 
                        t_em["ang"], 
                        t_em["neu"], 
                        t_em["sad"],
                        interaction
                    ]

    df_interaction.to_json(f"{dir_path}/data/df_interaction.json")
    