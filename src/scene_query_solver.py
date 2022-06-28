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
    df_interaction_generell = pd.DataFrame(data=d)

    for movie in movie_list:
        serialization_vision = SerializationMiddleware(JSONStorage)
        serialization_vision.register_serializer(DateTimeSerializer(), 'TinyDate')

        serialization_audio = SerializationMiddleware(JSONStorage)
        serialization_audio.register_serializer(DateTimeSerializer(), 'TinyDate')

        serialization_video = SerializationMiddleware(JSONStorage)
        serialization_video.register_serializer(DateTimeSerializer(), 'TinyDate')

        shot_keyframes = f"{hlvu_test}/keyframes/shot_keyf/{movie}"

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
        for scene in os.listdir(shot_keyframes):
                for shot in sorted(os.listdir(f"{shot_keyframes}/{scene}")):
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

        df_interaction_generell = df_interaction_generell.append(df_interaction, ignore_index=True)

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
            df_query = sqldf(f"SELECT * FROM df WHERE scene='{scene}'", locals())
            scene_interaction.append(list(df_query["interaction"].unique()))

        path = f"{hlvu_test}/Queries/Scene-level/{movie}.Scene-Level.Queries.xml"
        with open(path) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())

        root = minidom.Document()
  
        body = root.createElement('DeepVideoUnderstandingSceneResults') 
        body.setAttribute('movie', movie)
        root.appendChild(body)

        xml = root.createElement('DeepVideoUnderstandingRunResult') 
        xml.setAttribute('desc', "A Multi-Stream Approach for Video Understanding")
        xml.setAttribute('pid', "UZH")
        xml.setAttribute('priority', "1")
        body.appendChild(xml)


        question = list(data_dict["DeepVideoUnderstandingSceneQueries"].values())[1]

        for q in question:

            if (q["@question"] == "5") or (q["@question"] == "6"):
                pass
            else:
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
                    top = [i for i in sorted(sim_lst, reverse=True) if i!= 0]
                    if len(top) > 0:
                        for idx, similarity in enumerate(top):
                            scene = scenes[sim_lst.index(similarity)]
                            if sum(top) > 0:
                                confidence = similarity/sum(top) * 100
                            else:
                                confidence = 0
                            item = root.createElement('item')
                            item.setAttribute('order', str(idx+1))
                            item.setAttribute('scene', scene)
                            item.setAttribute('confidence', str(confidence))
                            productChild.appendChild(item)
                    else:
                        item = root.createElement('item')
                        item.setAttribute('order', "1")
                        item.setAttribute('scene', "None")
                        item.setAttribute('confidence', "0")
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
                            df_query1 = sqldf(
                                f"SELECT * FROM df WHERE scene='{scene}' AND person1 = '{obj}' AND interaction='{predicate}'", locals())
                            df_query2 = sqldf(
                                f"SELECT * FROM df WHERE scene='{scene}' AND person2 = '{obj}'AND interaction='{predicate}'", locals())
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
                    for answer in q["Answers"]["item"]:
                        if answer["@answer"] in relation_list: 
                            answers.append(answer["@answer"])
                    for item in q["item"]:
                        #print(item.keys())
                        if "@subject" in list(item.keys()):
                            #subj = item["@subject"].split(":")[1]
                            obj = item["@object"].split(":")[1]
                            predicate = item["@predicate"].split(":")[1]
                            if predicate in relation_list:
                                scene_nr = item["@scene"]
                                scene = f"{movie}-{scene_nr}"
                                df_query = sqldf(
                                    f"SELECT * FROM df WHERE scene='{scene}'", locals())
                                if len(df_query) > 0:
                                    ids = df_query["id"]
                                    prob_lst = []
                                    probs = []
                                    answer_not_found = True
                                    for i in ids:
                                        prob_lst.append(y_proba[i])
                                    predication_idx = relation_list.index(predicate)
                                    for prob in prob_lst:
                                        probs.append(prob[predication_idx])
                                    idx = probs.index(max(probs)) + 1
                                    if len(prob_lst) <= idx and len(probs) > 1:
                                        probs_sorted = probs.sort()
                                        max_second_val = probs_sorted[-2]
                                        idx  = probs.index(max_second_val) + 1
                                    if len(prob_lst) > idx:
                                        while answer_not_found:
                                            proba = prob_lst[idx]
                                            proba_list = []
                                            for answer in answers:
                                                rel_ind = relation_list.index(answer)
                                                if proba[rel_ind] > 0:
                                                    answer_not_found = False
                                                    proba_list.append(proba[rel_ind])
                                            idx += 1
                                            if answer_not_found == False:
                                                max_val = max(proba_list)
                                                proba_lst = list(proba)
                                                answer_idx = proba_lst.index(max_val)
                                                item = root.createElement('item')
                                                item.setAttribute('type', "Interaction")
                                                item.setAttribute('answer', relation_list[answer_idx])
                                                productChild.appendChild(item)
                                            elif idx == len(prob_lst):
                                                answer_not_found = False
                                                #item = root.createElement('item')
                                                #item.setAttribute('type', "Interaction")
                                                #item.setAttribute('answer', "relation not found")
                                                #productChild.appendChild(item)
                                    #else:
                                    #    item = root.createElement('item')
                                    #    item.setAttribute('type', "Interaction")
                                    #    item.setAttribute('answer', "only one interaction available for this scene")
                                    #    productChild.appendChild(item)
                            #else:
                            #    item = root.createElement('item')
                            #    item.setAttribute('type', "Interaction")
                            #    item.setAttribute('answer', "subject not in list")
                            #    productChild.appendChild(item)

                        
                if query_type == "4":
                    answers = []
                    for answer in q["Answers"]["item"]:
                        if answer["@answer"] in relation_list:
                            answers.append(answer["@answer"])
                    for item in q["item"]:
                        if "@subject" in list(item.keys()):
                            #subj = item["@subject"].split(":")[1]
                            obj = item["@object"].split(":")[1]
                            predicate = item["@predicate"].split(":")[1]
                            if predicate in relation_list:
                                scene_nr = item["@scene"]
                                scene = f"{movie}-{scene_nr}"
                                df_query = sqldf(
                                    f"SELECT * FROM df WHERE scene='{scene}'", locals())
                                if len(df_query) > 0:
                                    ids = df_query["id"]
                                    prob_lst = []
                                    probs = []
                                    answer_not_found = True
                                    for i in ids:
                                        prob_lst.append(y_proba[i])
                                    predication_idx = relation_list.index(predicate)
                                    for prob in prob_lst:
                                        probs.append(prob[predication_idx])
                                    idx = probs.index(max(probs)) - 1
                                    if len(prob_lst) <= idx:
                                        max_second_val = probs.sort()[-2]
                                        idx  = probs.index(max_second_val) - 1
                                    if len(prob_lst) <= idx and len(probs) > 1:
                                        probs_sorted = probs.sort()
                                        max_second_val = probs_sorted[-2]
                                        idx  = probs.index(max_second_val) - 1
                                    if len(prob_lst) > idx:
                                        while answer_not_found:
                                            proba = prob_lst[idx]
                                            proba_list = []
                                            for answer in answers:
                                                rel_ind = relation_list.index(answer)
                                                if proba[rel_ind] > 0:
                                                    answer_not_found = False
                                                    proba_list.append(proba[rel_ind])
                                            idx += 1
                                            if answer_not_found == False:
                                                max_val = max(proba_list)
                                                proba_lst = list(proba)
                                                answer_idx = proba_lst.index(max_val)
                                                item = root.createElement('item')
                                                item.setAttribute('type', "Interaction")
                                                item.setAttribute('answer', relation_list[answer_idx])
                                                productChild.appendChild(item)
                                            elif idx == len(prob_lst):
                                                answer_not_found = False
                                            #    item = root.createElement('item')
                                            #    item.setAttribute('type', "Interaction")
                                            #    item.setAttribute('answer', "None")
                                            #    productChild.appendChild(item)
                                    #else:
                                    #    item = root.createElement('item')
                                    #    item.setAttribute('type', "Interaction")
                                    #    item.setAttribute('answer', "None")
                                    #    productChild.appendChild(item)
                            #else:
                            #    item = root.createElement('item')
                            #    item.setAttribute('type', "Interaction")
                            #    item.setAttribute('answer', "None")
                            #    productChild.appendChild(item)
                
                #if query_type == "5":
                #    pass


                #if query_type == "6":
                #    pass

                            

        xml_str = root.toprettyxml(indent ="\t") 
        if not os.path.exists(f"{dir_path}/submissions/scene"):
            if not os.path.exists(f"{dir_path}/submissions"):
                os.mkdir(f"{dir_path}/submissions")
                os.mkdir(f"{dir_path}/submissions/dir_path")
            else:
                os.mkdir(f"{dir_path}/submissions/scene")
        save_path_file = f"{dir_path}/submissions/scene/{movie}_scene.xml"
        with open(save_path_file, "w") as f:
            f.write(xml_str)

        df_interaction_generell.to_json(f"{dir_path}/data/df_interaction_test.json")