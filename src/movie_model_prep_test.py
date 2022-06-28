import pandas as pd
import itertools
from tqdm import tqdm
from relationship_helper.relation import *
from tinydb.storages import JSONStorage
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer
from settings import HLVU_LOCATION



def create_data(hlvu_location, dir_path, movie_list, answer_path_exists = False):

    d = {
        'person1': [], 
        'person2': [],
        'movie':[],
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
        'movie':[],
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

    #d = {
    #    'concept': [], 
    #    'person': [],
    #    'movie':[],
    #    'concept_stat': [], 
    #    'polarity': [], 
    #    'voice_angry':[],
    #    'voice_happy':[],
    #    'voice_neutral':[],
    #    'voice_sad':[], 
    #    'relation':[]
    #}

    #movie_dfpc = pd.DataFrame(data=d)

    action_classes = get_action_classes(dir_path)
    location_classes = get_location_classes(dir_path)

    for movie in movie_list:
        path = f"{hlvu_location}/keyframes/shot_keyf/{movie}"

        serialization_vision = SerializationMiddleware(JSONStorage)
        serialization_vision.register_serializer(DateTimeSerializer(), 'TinyDate')

        serialization_audio = SerializationMiddleware(JSONStorage)
        serialization_audio.register_serializer(DateTimeSerializer(), 'TinyDate')

        serialization_video = SerializationMiddleware(JSONStorage)
        serialization_video.register_serializer(DateTimeSerializer(), 'TinyDate')


        vision_db = TinyDB(f'{dir_path}/database/vision_{movie}.json', storage=serialization_vision)

        video_db = TinyDB(f'{dir_path}/database/action_test.json', storage=serialization_video)

        audio_db = TinyDB(f'{dir_path}/database/audio_{movie}.json', storage=serialization_audio)

        entity_type ={}
        data=pd.read_table(f"{hlvu_location}/Queries/movie_knowledge_graph/{movie}/{movie}.entity.types.txt", delimiter = ':', header= None)

        for _,row in data.iterrows():
            entity_type[row[0].replace(" ","").lower()] = row[1].replace(" ","")

        if answer_path_exists:
            f = open(f"{hlvu_location}/Queries/movie_knowledge_graph/{movie}/{movie}.Movie-level.txt", "r")
            lines = [i.strip("\n") for i in f.readlines()]

            df = pd.read_excel(f"{HLVU_LOCATION}/movie_knowledge_graph/HLVU_Relationships_Definitions.xlsx")

            relation_dict = {}
            for _,row in df.iterrows():
                relation_dict[row["Inverse"]] = row["Relation"]

            entity_1 = []
            entity_2 = []
            relationship = []
            for line in lines:
                line_parsed = line.split("-.->")
                if len(line_parsed) > 1:
                    non_space = [i.strip(" ") for i in line_parsed]
                    entities = non_space[0::2]
                    relation = non_space[1::2]
                    for x,y,z in zip(range(0,len(entities)-1),range(1, len(entities)), range(0, len(relation))):
                        if relation[z] in list(relation_dict.values()):
                            entity_1.append(entities[x].lower())
                            entity_2.append(entities[y].lower())
                            relationship.append(relation[z])
                        else:
                            if relation[z] in list(relation_dict.keys()):
                                inverse_relation = relation_dict[relation[z]]
                                entity_1.append(entities[y].lower())
                                entity_2.append(entities[x].lower())
                                relationship.append(inverse_relation)

            df = pd.DataFrame(list(zip(entity_1, entity_2, relationship)),
            columns =['entity_1', 'entity_2', "relation"])

            relation_d = {}

            df_relation = df.drop_duplicates()
            for _, row in df_relation.iterrows():
                relation_d[(row["entity_1"], row["entity_2"])] = row["relation"]


        t1 = tuple(entity_type.keys())
        combis_pp = []
        for i in itertools.product(t1,t1):
            if i[0] != i[1] and ((i[1], i[0]) not in combis_pp):
                if entity_type[i[0]] == "Person" and entity_type[i[1]] == "Person":
                    combis_pp.append((i[0], i[1]))

        for i in tqdm(combis_pp):
            scene_stat, shot_stat, text_stat, emotions, places365, action, sentiment = get_stats(i[0], i[1], movie, path, vision_db, video_db, audio_db, location_classes, action_classes)
            movie_dfpp.loc[movie_dfpp.shape[0]] = [
                i[0], i[1], movie,
                scene_stat, 
                shot_stat, 
                emotions["angry"], emotions["fear"], emotions["neutral"], emotions["sad"], emotions["suprise"],
                action[0][0], action[0][1], action[1][0], action[1][1], action[2][0], action[2][1],
                places365[0][0], places365[0][1], places365[1][0], places365[1][1], places365[2][0], places365[2][1], 
                sentiment, 
                text_stat
                , None
                ]
            """
            except:
                scene_stat, shot_stat, text_stat, emotions, places365, action, sentiment = get_stats(i[0], i[1], movie, path,vision_db, video_db, audio_db, location_classes, action_classes)
                movie_dfpp.loc[movie_dfpp.shape[0]] = [
                    i[0], i[1], movie,
                    scene_stat, 
                    shot_stat,
                    emotions["angry"], emotions["fear"], emotions["neutral"], emotions["sad"], emotions["suprise"],
                    action[0][0], action[0][1], action[1][0], action[1][1], action[2][0], action[2][1],
                    places365[0][0], places365[0][1], places365[1][0], places365[1][1], places365[2][0], places365[2][1], 
                    sentiment, 
                    text_stat, 
                    None
                ]
            """

        t1 = tuple(entity_type.keys())
        combis_pl = []
        
        for i in itertools.product(t1,t1):
            if i[0] != i[1] and ((i[1], i[0]) not in combis_pl):
                if (entity_type[i[0]] == "Location" and entity_type[i[1]] == "Person"):
                    combis_pl.append((i[0], i[1]))

        for i in tqdm(combis_pl):
            #try:
            delf, places365 = get_person_location_stats(i[0], i[1], vision_db, location_classes)
            movie_dfpl.loc[movie_dfpl.shape[0]] = [
                i[0], i[1], movie,
                delf, 
                places365[0][0], places365[0][1], places365[1][0], places365[1][1], places365[2][0], places365[2][1],
                None
            ]
            """
            except:
                delf, places365 = get_person_location_stats(i[0], i[1], vision_db, location_classes)
                movie_dfpl.loc[movie_dfpl.shape[0]] = [
                    i[0], i[1], movie,
                    delf, 
                    places365[0][0], places365[0][1], places365[1][0], places365[1][1], places365[2][0], places365[2][1],
                    None
                ]
            """

        """
        t1 = tuple(entity_type.keys())
        combis_pc = []

        
        for i in itertools.product(t1,t1):
            if i[0] != i[1] and ((i[1], i[0]) not in combis_pc):
                if (entity_type[i[0]] == "Person" and entity_type[i[1]] == "Concept"):
                    combis_pc.append((i[0], i[1]))

        for i in tqdm(combis_pc):
            try:
                concept_stat, average_polarity, emotion_list = get_person_concept_stats(i[0], i[1], audio_db)
                movie_dfpc.loc[movie_dfpc.shape[0]] = [
                    i[0], i[1], movie,
                    concept_stat, 
                    average_polarity, 
                    emotion_list["ang"], emotion_list["hap"], emotion_list["neu"], emotion_list["sad"], 
                    relation_d[i]
                ]
            except:
                concept_stat, average_polarity, emotion_list = get_person_concept_stats(i[0], i[1], audio_db)
                movie_dfpc.loc[movie_dfpc.shape[0]] = [
                    i[0], i[1], movie,
                    concept_stat, 
                    average_polarity, 
                    emotion_list["ang"], emotion_list["hap"], emotion_list["neu"], emotion_list["sad"], 
                    None
                ]
        """
    
    if not os.path.exists(f"{dir_path}/data"):
        os.mkdir(f"{dir_path}/data")

    movie_dfpp.to_json(f'{dir_path}/data/people2people_test.json')
    movie_dfpl.to_json(f'{dir_path}/data/people2location_test.json')
    #movie_dfpc.to_json(f'{dir_path}/data/people2concept_test.json')
