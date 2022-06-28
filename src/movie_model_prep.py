import pandas as pd
from tinydb import TinyDB,Query,where
from datetime import datetime
from tinydb.storages import JSONStorage
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer
import itertools
from relationship_helper.relation import *
from settings import HLVU_LOCATION, DIR_PATH
from tqdm import tqdm
import itertools


def create_dataframe(movie_list, dir_path, hlvu_location):

    d = {
        'person1': [], 
        'person2': [],
        'shotlevel': [], 
        'scenelevel': [], 
        'face_angry' : [],
        'face_fear' : [],
        'face_neutral' : [],
        'face_sad' : [],
        'face_surprise' : [], 
        'action_1': [],
        'action_1_freq': [],
        'action_2': [],
        'action_2_freq': [],
        'action_3': [],
        'action_3_freq': [], 
        'places365_1' : [],
        'places365_1_freq' : [],
        'places365_2' : [],
        'places365_2_freq' : [],
        'places365_3' : [],
        'places365_3_freq' : [], 
        'text_sentiment':[], 
        'text_level':[], 
        'relation':[]
        }

    movie_dfpp = pd.DataFrame(data=d)

    d = {
        'location': [], 
        'person': [],
        'delf': [], 
        'places365_1' : [],
        'places365_1_freq' : [],
        'places365_2' : [],
        'places365_2_freq' : [],
        'places365_3' : [],
        'places365_3_freq' : [],
        'relation':[]
        }

    movie_dfpl = pd.DataFrame(data=d)

    d = {
        'concept': [], 
        'person': [],
        'concept_stat': [], 
        'polarity': [], 
        'voice_angry':[],
        'voice_happy':[],
        'voice_neutral':[],
        'voice_sad':[], 
        'relation':[]
        }

    movie_dfpc = pd.DataFrame(data=d)

    action_classes = get_action_classes(dir_path)
    location_classes = get_location_classes(dir_path)

    for movie in movie_list:

        person_d = {}
        relation_d = {}
        entity_type ={}


        
        file = f"{hlvu_location}/movie_knowledge_graph/{movie}/{movie}.entity.types.txt"
        data=pd.read_table(file, delimiter = ':', header= None)

        for _,row in data.iterrows():
            entity_type[row[0].replace(" ","")] = row[1].replace(" ","")
        
        f = open(f"{hlvu_location}/movie_knowledge_graph/{movie}/{movie}.tgf", "r")
        relation_change = False

        for line in f.readlines():
            if not relation_change:
                split_lines = line.strip("\n").split(" ")
                if split_lines[0] == "#":
                    relation_change = True
                else:
                    value = ''.join(split_lines[1:])
                    person_d[split_lines[0]] = value
            else:
                split_lines = line.strip("\n").split(" ")
                #print(split_lines)
                relation_d[(split_lines[0],split_lines[1])] = split_lines[2:]

        
        t1 = tuple(person_d.keys())
        combis_pp = []
        for i in itertools.product(t1,t1):
            if i[0] != i[1] and ((i[1], i[0]) not in combis_pp):
                if not (entity_type[person_d[i[0]]] == "Location" or entity_type[person_d[i[1]]] == "Location"):
                    combis_pp.append((i[0], i[1]))

        path = f"{hlvu_location}/keyframes/shot_keyf/{movie}"

        serialization_vision = SerializationMiddleware(JSONStorage)
        serialization_vision.register_serializer(DateTimeSerializer(), 'TinyDate')

        serialization_audio = SerializationMiddleware(JSONStorage)
        serialization_audio.register_serializer(DateTimeSerializer(), 'TinyDate')

        serialization_video = SerializationMiddleware(JSONStorage)
        serialization_video.register_serializer(DateTimeSerializer(), 'TinyDate')


        vision_db = TinyDB(f'{dir_path}/database/vision_{movie}.json', storage=serialization_vision)
        video_db = TinyDB(f'{dir_path}/database/action.json', storage=serialization_video)
        audio_db = TinyDB(f'{dir_path}/database/audio_{movie}.json', storage=serialization_audio)

        df_relations = pd.read_excel(f'{hlvu_location}/movie_knowledge_graph/HLVU_Relationships_Definitions.xlsx')
        relation_dict = {}

        for _,row in df_relations.iterrows():
            relation_dict[row["Relation"].lower()] = row["Inverse"].lower()

        for i in tqdm(combis_pp):
            #print(person_d[i[0]],person_d[i[1]], i)
            j = (i[1],i[0])
            if i in relation_d.keys() or j in relation_d.keys():
                if i in relation_d.keys():
                    #print(' '.join(relation_d[i]))
                    scene_stat, shot_stat, text_stat, emotions, places365, action, sentiment = get_stats(person_d[i[0]], person_d[i[1]], movie, path, vision_db, video_db, audio_db, location_classes, action_classes)
                    movie_dfpp.loc[movie_dfpp.shape[0]] = [
                        person_d[i[0]], person_d[i[1]], 
                        scene_stat, 
                        shot_stat, 
                        emotions["angry"], emotions["fear"], emotions["neutral"], emotions["sad"], emotions["suprise"],
                        action[0][0], action[0][1], action[1][0], action[1][1], action[2][0], action[2][1],
                        places365[0][0], places365[0][1], places365[1][0], places365[1][1], places365[2][0], places365[2][1], 
                        sentiment, 
                        text_stat
                        ,' '.join(relation_d[i])
                    ]
            
                if j in relation_d.keys():
                    #print(' '.join(relation_d[j]))
                    scene_stat, shot_stat, text_stat, emotions, places365, action, sentiment = get_stats(person_d[j[0]], person_d[j[1]], movie, path, vision_db, video_db, audio_db, location_classes, action_classes)
                    movie_dfpp.loc[movie_dfpp.shape[0]] = [
                        person_d[j[0]], person_d[j[1]], 
                        scene_stat, 
                        shot_stat, 
                        emotions["angry"], emotions["fear"], emotions["neutral"], emotions["sad"], emotions["suprise"],
                        action[0][0], action[0][1], action[1][0], action[1][1], action[2][0], action[2][1],
                        places365[0][0], places365[0][1], places365[1][0], places365[1][1], places365[2][0], places365[2][1], 
                        sentiment, 
                        text_stat
                        ,' '.join(relation_d[j])
                ]
            
            else:
                #print("None")
                scene_stat, shot_stat, text_stat, emotions, places365, action, sentiment = get_stats(person_d[i[0]], person_d[i[1]], movie, path,vision_db, video_db, audio_db, location_classes, action_classes)
                movie_dfpp.loc[movie_dfpp.shape[0]] = [
                    person_d[i[0]], person_d[i[1]],
                    scene_stat, 
                    shot_stat,
                    emotions["angry"], emotions["fear"], emotions["neutral"], emotions["sad"], emotions["suprise"],
                    action[0][0], action[0][1], action[1][0], action[1][1], action[2][0], action[2][1],
                    places365[0][0], places365[0][1], places365[1][0], places365[1][1], places365[2][0], places365[2][1], 
                    sentiment, 
                    text_stat, 
                    None
                ]

        t1 = tuple(person_d.keys())
        combis_pl = []
        for i in itertools.product(t1,t1):
            if i[0] != i[1] and ((i[1], i[0]) not in combis_pl):
                if (entity_type[person_d[i[0]]] == "Location" and entity_type[person_d[i[1]]] == "Person"):
                    combis_pl.append((i[0], i[1]))

        for i in tqdm(combis_pl):

            j = (i[1],i[0])
            if i in relation_d.keys() or j in relation_d.keys():
                if i in relation_d.keys():
                    delf, places365 = get_person_location_stats(person_d[i[0]], person_d[i[1]], vision_db, location_classes)
                    movie_dfpl.loc[movie_dfpl.shape[0]] = [
                        person_d[i[0]], person_d[i[1]], 
                        delf, 
                        places365[0][0], places365[0][1], places365[1][0], places365[1][1], places365[2][0], places365[2][1],
                        ' '.join(relation_d[i])
                    ]

                if j in relation_d.keys():
                    delf, places365 = get_person_location_stats(person_d[j[0]], person_d[j[1]], vision_db, location_classes)
                    movie_dfpl.loc[movie_dfpl.shape[0]] = [
                        person_d[j[0]], person_d[j[1]], 
                        delf, 
                        places365[0][0], places365[0][1], places365[1][0], places365[1][1], places365[2][0], places365[2][1],
                        ' '.join(relation_d[j])
                    ]
            
            else:
                delf, places365 = get_person_location_stats(person_d[i[0]], person_d[i[1]], vision_db, location_classes)
                movie_dfpl.loc[movie_dfpl.shape[0]] = [
                    person_d[i[0]], person_d[i[1]],
                    delf, 
                    places365[0][0], places365[0][1], places365[1][0], places365[1][1], places365[2][0], places365[2][1],
                    None
                ]

        t1 = tuple(person_d.keys())
        combis_pc = []

        for i in itertools.product(t1,t1):
            if i[0] != i[1] and ((i[1], i[0]) not in combis_pc):
                if (entity_type[person_d[i[0]]] == "Person" and entity_type[person_d[i[1]]] == "Concept"):
                    combis_pc.append((i[0], i[1]))

        for i in tqdm(combis_pc):
            try:
                concept_stat, average_polarity, emotion_list = get_person_concept_stats(person_d[i[0]], person_d[i[1]], audio_db)
                movie_dfpc.loc[movie_dfpc.shape[0]] = [
                    person_d[i[0]], person_d[i[1]], 
                    concept_stat, 
                    average_polarity, 
                    emotion_list["ang"], emotion_list["hap"], emotion_list["neu"], emotion_list["sad"], 
                    ' '.join(relation_d[i])
                ]
            except:
                concept_stat, average_polarity, emotion_list = get_person_concept_stats(person_d[i[0]], person_d[i[1]], audio_db)
                movie_dfpc.loc[movie_dfpc.shape[0]] = [
                    person_d[i[0]], person_d[i[1]],
                    concept_stat, 
                    average_polarity, 
                    emotion_list["ang"], emotion_list["hap"], emotion_list["neu"], emotion_list["sad"], 
                    None
                ]
    
    if not os.path.exists(f"{dir_path}/data"):
        os.mkdir(f"{dir_path}/data")
    
    movie_dfpp.to_json(f'{dir_path}/data/people2people.json')
    movie_dfpl.to_json(f'{dir_path}/data/people2location.json')
    movie_dfpc.to_json(f'{dir_path}/data/people2concept.json')
