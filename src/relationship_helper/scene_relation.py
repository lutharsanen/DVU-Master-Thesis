import json
import pandas as pd
from tinydb import TinyDB,Query,where

def prep_data_creator(data_path):
 
    # Opening JSON file
    f = open(data_path)

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    nodes = data["nodes"]
    links = data["links"]

    node_dict = {}

    for node in nodes:
        node_dict[node["key"]] = (node["text"],node["type"])

    data = []

    for link in links:
        data.append([node_dict[link["from"]],node_dict[link["to"]]])

    struc_data = []
    first_array= []

    for elements in data:
        first_array.append(elements)
        if "Sequence" in elements[0] or "Sequence" in elements[1]:
            struc_data.append(first_array)
            first_array = []

    location = []
    location_person = []
    location_sequence = []

    interaction = []
    interaction_person_1 = []
    interaction_person_2 = []

    sequence = []

    emotion = []
    emotion_person = []
    emotion_sequence = []


    for idx, seq in enumerate(struc_data):
        for point in seq:
            if point[0][1] == "Location":
                location.append(point[0][0])
                location_person.append(point[1][0])
                location_sequence.append(idx)

            elif point[1][1] == "Location":
                location.append(point[1][0])
                location_person.append(point[0][0])
                location_sequence.append(idx)

            if point[0][1] == "Emotion":
                emotion.append(point[0][0])
                emotion_person.append(point[1][0])
                emotion_sequence.append(idx)

            elif point[1][1] == "Emotion":
                emotion.append(point[1][0])
                emotion_person.append(point[0][0])
                emotion_sequence.append(idx)

            if point[0][1] == "Interaction" and point[1][1] == "Person":
                if point[0][0] not in interaction:
                    interaction.append(point[0][0])
                    interaction_person_1.append(point[1][0])

                else:
                    interaction_person_2.append(point[1][0])

            elif point[1][1] == "Interaction" and point[0][1] == "Person":
                if point[1][0] not in interaction:
                    interaction.append(point[1][0])
                    interaction_person_1.append(point[0][0])
                else:
                    interaction_person_2.append(point[0][0])

            if point[1][1] == "Interaction" and point[0][1] == "Sequence":
                if len(interaction) > 0:
                    if interaction[-1] != point[1][0]:
                        interaction.append(point[1][0])
                        interaction_person_1.append(interaction_person_1[interaction.index(point[1][0])])
                        interaction_person_2.append(interaction_person_2[interaction.index(point[1][0])])

            if point[1][1] == "Sequence":
                sequence.append(point[1][0])
            elif point[0][1] == "Sequence":
                sequence.append(point[0][0])

    prep_interaction = pd.DataFrame(list(
        zip(interaction, interaction_person_1, interaction_person_2, sequence)),
                     columns =['action', 'person1', 'person2', 'sequence'])
    
    prep_emotion = pd.DataFrame(list(
        zip(emotion, emotion_person, emotion_sequence)),
                     columns =['emotion', 'person', 'sequence'])
    
    prep_location = pd.DataFrame(list(
        zip(location, location_person, location_sequence)),
                     columns =['location', 'person', 'sequence'])
    
    return prep_interaction, prep_emotion, prep_location

def vision_query(person_1, person_2, scene, shot, vision_db):
    User = Query()
    query = vision_db.search(
        User.faces.any([person_1, person_2]) 
        & (where('scene') == scene) 
        & (where('shots') == shot))
    
    return query

def action_query(scene, shot, video_db):
    User = Query()
    shot = f"{shot}.mp4"
    query = video_db.search(
        (User["scene"]==scene) 
        & (where("shot_name") == shot))
    
    return query

def audio_query(scene, start, end, audio_db):
    User = Query()
    scene = f"{scene}.wav"
    User = Query()
    query = audio_db.search(
        (User["scene"]==scene) 
        & (User.start > start) 
        & (User.end < end))
    
    return query

import os
import numpy as np
from collections import Counter

def get_emo_hist(lst):
    counter = Counter(lst)
    return counter



def get_scene_features(person1, person2, scene, split_list, kinetics400_to_interaction, vision_db, video_db, audio_db):

    transformed_action_lst = []
    emo_lst = []
    text_emo_lst = []
    text_lst = []
    for shot in split_list:
        emotions = []
        ####### vision ###########
        vis_query = vision_query(person1, person2, scene, shot, vision_db)
        if len(vis_query) > 0:
            for resp in vis_query:
                emotions.append(resp["emotions"][0])
            emo = get_emo_hist(emotions)
        ####### video ###########
            vid_query = action_query(scene, shot, video_db)
            action = vid_query[0]["action"]
            start = vid_query[0]["start_time"]
            end = vid_query[0]["end_time"]
            if action in kinetics400_to_interaction.keys():
                transformed_action = kinetics400_to_interaction[action]
            else:
                transformed_action = "unknown"
        ####### audio ##########
            aud_query = audio_query(scene, start, end, audio_db )
            if len(aud_query) > 0:
                text_emo = get_emo_hist(aud_query[0]["emotion"])
                text = aud_query[0]["text"]
                transformed_action_lst.append(transformed_action)
                emo_lst.append(emo)
                text_emo_lst.append(text_emo)
                text_lst.append(text)
        ########################
    return transformed_action_lst, emo_lst, text_emo_lst, text_lst



def get_test_scene_features(person1, person2, scene, shot, kinetics400_to_interaction, vision_db, video_db, audio_db):
    transformed_action_lst = []
    emo_lst = []
    text_emo_lst = []
    text_lst = []
    emotions = []
    ####### vision ###########
    vis_query = vision_query(person1, person2, scene, shot, vision_db)
    if len(vis_query) > 0:
        if len(vis_query) > 0:
            for resp in vis_query:
                if person1 in resp["faces"]:
                    index = resp["faces"].index(person1)
                    emotions.append(resp["emotions"][index])
                elif person2 in vis_query:
                    index = resp["faces"].index(person2)
                    emotions.append(resp["emotions"][index])
            emo = get_emo_hist(emotions)
            ####### video ###########
            vid_query = action_query(scene, shot, video_db)
            if len(vid_query) > 0:
                action = vid_query[0]["action"]
                start = vid_query[0]["start_time"]
                end = vid_query[0]["end_time"]
                if action in kinetics400_to_interaction.keys():
                    transformed_action = kinetics400_to_interaction[action]
                else:
                    transformed_action = "unknown"
                ####### audio ##########
                aud_query = audio_query(scene, start, end, audio_db )
                if len(aud_query) > 0:
                    text_emo = get_emo_hist(aud_query[0]["emotion"])
                    text = aud_query[0]["text"]
                    transformed_action_lst.append(transformed_action)
                    emo_lst.append(emo)
                    text_emo_lst.append(text_emo)
                    text_lst.append(text)
                ########################
            else:
                return 0
        else:
            return 0
        if len(emo_lst) == 0:
            return 0       
        return transformed_action_lst, emo_lst, text_emo_lst, text_lst
    else:
        return 0

def most_frequent(List):
    return max(set(List), key = List.count)