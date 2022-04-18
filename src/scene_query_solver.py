from relationship_helper.scene_relation import get_test_scene_features,action_query,audio_query,vision_query,prep_data_creator, most_frequent
from tinydb import TinyDB,Query,where
from tinydb.storages import JSONStorage
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer
from tqdm import tqdm
import pandas as pd
import os
from itertools import combinations
import joblib
from pandasql import sqldf
import xml.etree.ElementTree as ET
import xmltodict
import json
import xml.dom.minidom as minidom
from collections import Counter



def solve_query(dir_path, movie_list, hlvu_test, kinetics400_to_interaction):
    for movie in movie_list:
        serialization_vision = SerializationMiddleware(JSONStorage)
        serialization_vision.register_serializer(DateTimeSerializer(), 'TinyDate')

        serialization_audio = SerializationMiddleware(JSONStorage)
        serialization_audio.register_serializer(DateTimeSerializer(), 'TinyDate')

        serialization_video = SerializationMiddleware(JSONStorage)
        serialization_video.register_serializer(DateTimeSerializer(), 'TinyDate')


        vision_db = TinyDB(f"{dir_path}/database/vision_{movie}.json", storage=serialization_vision)
        video_db = TinyDB(f"{dir_path}/database/action_test.json", storage=serialization_video)
        audio_db = TinyDB(f"{dir_path}/database/audio_{movie}.json", storage=serialization_audio)

        d = {
            'person1': [], 
            'person2': [],
            'scene':[],
            'shot':[],
            'action': [], 
            'face_happy':[], 
            'face_angry':[], 
            'face_neutral':[],
            'face_sad':[],
            'face_surprise':[],
            'text_happy':[],
            'text_angry':[],
            'text_neutral':[],
            'text_sad':[]
            }
    
        df_interaction = pd.DataFrame(data=d)
        User = Query()
        for scene in os.listdir(hlvu_test):
            for shot in sorted(os.listdir(f"{hlvu_test}/{scene}")):
                faces = []
                query = vision_db.search((User["scene"] == scene) & (where("shots") == shot))
                for res in query:
                    for face in res["faces"]:
                        if "unknown" not in face:
                            faces.append(face)
                if len(faces) == 1:
                    query = get_test_scene_features(faces[0], faces[0], scene, shot, kinetics400_to_interaction, vision_db, video_db, audio_db)
                    if query != 0:
                        df_interaction.loc[df_interaction.shape[0]] = [
                            faces[0], faces[0], scene, shot, query[0][0],
                            query[1][0]["happy"], 
                            query[1][0]["angry"], 
                            query[1][0]["neutral"], 
                            query[1][0]["sad"], 
                            query[1][0]["surprise"],
                            query[2][0]["hap"], 
                            query[2][0]["ang"], 
                            query[2][0]["neu"], 
                            query[2][0]["sad"]
                        ]
                elif len(faces) == 2:
                    query = get_test_scene_features(faces[0], faces[1], scene, shot, kinetics400_to_interaction, vision_db, video_db, audio_db)
                    if query != 0:
                        df_interaction.loc[df_interaction.shape[0]] = [
                            faces[0], faces[1], scene, shot, query[0][0],
                            query[1][0]["happy"], 
                            query[1][0]["angry"], 
                            query[1][0]["neutral"], 
                            query[1][0]["sad"], 
                            query[1][0]["surprise"],
                            query[2][0]["hap"], 
                            query[2][0]["ang"], 
                            query[2][0]["neu"], 
                            query[2][0]["sad"]
                        ]
                elif len(list(set(faces))) > 2:
                    unique_faces = list(set(faces))
                    combos = combinations(unique_faces, 2)
                    for i in list(combos):
                        query = get_test_scene_features(i[0], i[1], scene, shot, kinetics400_to_interaction, vision_db, video_db, audio_db)
                        if query != 0:
                            df_interaction.loc[df_interaction.shape[0]] = [
                            i[0], i[1], scene, shot, query[0][0],
                            query[1][0]["happy"], 
                            query[1][0]["angry"], 
                            query[1][0]["neutral"], 
                            query[1][0]["sad"], 
                            query[1][0]["surprise"],
                            query[2][0]["hap"], 
                            query[2][0]["ang"], 
                            query[2][0]["neu"], 
                            query[2][0]["sad"]
                        ]

        X_test = df_interaction.drop(columns = ["person1", "person2", "scene", "shot"])
        kinetics_values = list(kinetics400_to_interaction.values())
        kinetics_values.append("unknown")
        action_list = [kinetics_values.index(i) for i in df_interaction["action"]]
        X_test["action"] = action_list
        clf = joblib.load(f"{dir_path}/models/interaction_classifier.sav")
        y_predicted = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        # define an empty list
        relation_list = []

        # open file and read the content in a list
        with open(f'{dir_path}/data/interactions.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                relation = line[:-1]

                # add item to the list
                relation_list.append(relation)

        y = [relation_list[i] for i in y_predicted]

        df = pd.DataFrame(zip(
            [i for i in range(len(df_interaction))],
            df_interaction["person1"], 
            df_interaction["person2"], 
            df_interaction["shot"], 
            df_interaction["scene"],
            y), columns = ["id", "person1", "person2", "shot", "scene", "interaction"]
        )

        scenes = sorted(list(df["scene"].unique()))
        scene_interaction = []

        for scene in scenes:
            mysql = lambda q: sqldf(q, globals())
            df_query = mysql(f"SELECT * FROM df WHERE scene='{scene}'")
            scene_interaction.append(list(df_query["interaction"].unique()))

        path = f"{hlvu_test}/Queries/Scene-level/{movie}.Scene_Level.xml"

        with open(path) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())

        root = minidom.Document()
  
        xml = root.createElement('DeepVideoUnderstandingSceneResult') 
        xml.setAttribute('scenes', movie)
        root.appendChild(xml)


        question = list(data_dict["DeepVideoUnderstandingSceneQueries"].values())[1]

        for q in question:
            query_type = q["@question"]
            query_id = q["@id"]
            productChild = root.createElement('DeepVideoUnderstandingTopicResult')
            productChild.setAttribute('question', query_type)
            productChild.setAttribute('id', query_id)
            xml.appendChild(productChild)
            
            
            if query_type == "1":
                interaction_answer = []
                for item in q["item"]:
                    if "@predicate" in item.keys():
                        interaction_answer.append(item["@predicate"].split(":")[1])
                sim_lst = []
                for interactions in scene_interaction:
                    res = len(set(interactions) & set(interaction_answer)) / float(len(set(interactions) | set(interaction_answer))) * 100
                    sim_lst.append(res)
                top4 = sorted(sim_lst[:4], reverse=True)
                for idx, similarity in enumerate(top4):
                    scene = scenes[sim_lst.index(similarity)]
                    if sum(top4) > 0:
                        confidence = similarity/sum(top4) * 100
                    else:
                        confidence = 0
                    item = root.createElement('item')
                    item.setAttribute('order', str(idx+1))
                    item.setAttribute('scene', scene)
                    item.setAttribute('confidence', str(confidence))
                    productChild.appendChild(item)
                    
                    
            if query_type =="2":
                xml.appendChild(productChild)
                answers = []
                for item in q["item"]:
                    if "@subject" in item.keys():
                        scene_nr = item["@scene"]
                        scene = f"{movie}-{scene_nr}"
                        predicate = item["@predicate"].split(":")[1]
                        obj = item["@object"].split(":")[1]
                        mysql = lambda q: sqldf(q, globals())
                        df_query1 = mysql(
                            f"SELECT * FROM df WHERE scene='{scene}' AND person1 = '{obj}' AND interaction='{predicate}'")
                        df_query2 = mysql(
                            f"SELECT * FROM df WHERE scene='{scene}' AND person2 = '{obj}'AND interaction='{predicate}'")
                        if len(df_query1) > 0:
                            answers.append(df_query1["person2"])
                        elif len(df_query2) > 0:
                            answers.append(df_query1["person1"])
                if len(answers) > 0:
                    counter = 1
                    for person,count in Counter(answers).most_common():
                        confidence = count/len(answers)
                        item = root.createElement('item')
                        item.setAttribute('order', str(counter))
                        item.setAttribute('subject', f"Person:{person}")
                        item.setAttribute('confidence', str(confidence))
                        productChild.appendChild(item)
                else:
                    item = root.createElement('item')
                    item.setAttribute('order', str(1))
                    item.setAttribute('subject', "Person:unknown")
                    item.setAttribute('confidence', str(0))
                    productChild.appendChild(item)

            if query_type =="3":
                answers = []
                query_answer = []
                for answer in q["Answers"]["item"]:
                    answers.append(answer["@answer"])
                for item in q["item"]:
                    if "@subject" in list(item.keys()):
                        subj = item["@subject"].split(":")[1]
                        obj = item["@object"].split(":")[1]
                        predicate = item["@predicate"].split(":")[1]
                        scene_nr = item["@scene"]
                        scene = f"{movie}-{scene_nr}"
                        df_query1 = mysql(
                            f"SELECT * FROM df WHERE scene='{scene}' AND person1 = '{obj}' AND person2 = '{subj}'")
                        df_query2 = mysql(
                            f"SELECT * FROM df WHERE scene='{scene}' AND person1 = '{subj}' AND person2 = '{obj}'")
                        if len(df_query1) > 0:
                            # get next column
                            next_value = False
                            while next_value == False:
                                idx = df_query1[df_query1["interaction"] == predicate].index +1
                                next_action = df_query1[idx]["interaction"]
                                next_id = df_query1.loc[idx]["id"]
                                if predicate != next_action:
                                    next_value = True
                            proba_lst = y_proba[next_id]
                            for answer in answers:
                                rel_ind = relation_list.index(answer)
                                proba_lst.append(relation_list[rel_ind])
                            final_answer = answers.index(max(proba_lst))
                            query_answer.append(final_answer)

                        elif len(df_query2) > 0:
                            next_value = False
                            while next_value == False:
                                idx = df_query1[df_query1["interaction"] == predicate].index +1
                                next_action = df_query1[idx]["interaction"]
                                next_id = df_query1.loc[idx]["id"]
                                if predicate != next_action:
                                    next_value = True
                            proba_lst = y_proba[next_id]
                            for answer in answers:
                                rel_ind = relation_list.index(answer)
                                proba_lst.append(relation_list[rel_ind])
                            final_answer = answers.index(max(proba_lst))
                            query_answer.append(final_answer)
                if len(query_answer) > 0:
                    query_result = most_frequent(query_answer)
                    item = root.createElement('item')
                    item.setAttribute('type', "Interaction")
                    item.setAttribute('answer', query_result)
                    productChild.appendChild(item)
                else:
                    item = root.createElement('item')
                    item.setAttribute('type', "Interaction")
                    item.setAttribute('answer', "unknown")
                    productChild.appendChild(item)
                    
            if query_type == "4":
                answers = []
                for answer in q["Answers"]["item"]:
                    answers.append(answer["@answer"])
                for item in q["item"]:
                    if "@subject" in list(item.keys()):
                        subj = item["@subject"].split(":")[1]
                        obj = item["@object"].split(":")[1]
                        predicate = item["@object"].split(":")[1]
                        scene_nr = item["@scene"]
                        scene = f"{movie}-{scene_nr}"
                        df_query1 = mysql(
                            f"SELECT * FROM df WHERE scene='{scene}' AND person1 = '{obj}' AND person2 = '{subj}'")
                        df_query2 = mysql(
                            f"SELECT * FROM df WHERE scene='{scene}' AND person1 = '{subj}' AND person2 = '{obj}'")
                        if len(df_query1) > 0:
                            # get previous column
                            next_value = False
                            while next_value == False:
                                idx = df_query1[df_query1["interaction"] == predicate].index-1
                                next_action = df_query1[idx]["interaction"]
                                next_id = df_query1.loc[idx]["id"]
                                if predicate != next_action:
                                    next_value = True
                            proba_lst = y_proba[next_id]
                            for answer in answers:
                                rel_ind = relation_list.index(answer)
                                proba_lst.append(relation_list[rel_ind])
                            final_answer = answers.index(max(proba_lst))
                            query_answer.append(final_answer)
                        elif len(df_query2) > 0:
                            next_value = False
                            while next_value == False:
                                idx = df_query1[df_query1["interaction"] == predicate].index-1
                                next_action = df_query1[idx]["interaction"]
                                next_id = df_query1.loc[idx]["id"]
                                if predicate != next_action:
                                    next_value = True
                            proba_lst = y_proba[next_id]
                            for answer in answers:
                                rel_ind = relation_list.index(answer)
                                proba_lst.append(relation_list[rel_ind])
                            final_answer = answers.index(max(proba_lst))
                            query_answer.append(final_answer)
                if len(query_answer) > 0:
                    query_result = most_frequent(query_answer)
                    item = root.createElement('item')
                    item.setAttribute('type', "Interaction")
                    item.setAttribute('answer', query_result)
                    productChild.appendChild(item)
                else:
                    item = root.createElement('item')
                    item.setAttribute('type', "Interaction")
                    item.setAttribute('answer', "unknown")
                    productChild.appendChild(item)

        xml_str = root.toprettyxml(indent ="\t") 
        if not os.path.exists(f"{dir_path}/submissions/scene"):
            os.mkdir(f"{dir_path}/submissions/scene")
        save_path_file = f"{dir_path}/submissions/scene/{movie}_scene.xml"
        with open(save_path_file, "w") as f:
            f.write(xml_str) 