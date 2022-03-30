from tinydb import TinyDB,Query,where
from textblob import TextBlob
import os
from collections import Counter
import json

def character_checker(name1, name2, shot_level, scene_level, vision_db, video_db):
    User = Query()
    emotions = []
    emotion_temp = []
    places365_features = []
    places_temp = []
    action = None
    test = vision_db.search(User.faces.any([name1, name2]) 
                            & (where('shots') == shot_level) 
                            & (where('scene') == scene_level))
    truth_check = [False, False]
    for result in test:
        if name1 in result["faces"]:
            truth_check[0] = True
            name_ind = result["faces"].index(name1)
            emotion_temp.append(result["emotions"][name_ind])
            places_temp.append(result["places365"])
        if name2 in result["faces"]:
            truth_check[1] = True
            name_ind = result["faces"].index(name2)
            emotion_temp.append(result["emotions"][name_ind])
            places_temp.append(result["places365"])
        #print(truth_check)
    outcome = all(truth_check)
    if outcome:
        emotions = emotion_temp
        places365_features = places_temp
        action = video_db.search((User['shot_name'] == f"{shot_level}.mp4") & (User['scene'] == scene_level))[0]["action"]
    return outcome, emotions, places365_features, action

def scene_character_checker(name1, name2, scene_level, vision_db, audio_db):
    User = Query()
    test = vision_db.search(User.faces.any([name1, name2]) & (where('scene') == scene_level))
    truth_check = [False, False]
    for result in test:
        if name1 in result["faces"]:
            truth_check[0] = True
        if name2 in result["faces"]:
            truth_check[1] = True
        #print(truth_check)
    outcome = all(truth_check)
    if outcome:
        count, sentiment = audio_analyzer(scene_level, name1, name2, audio_db)
    else:
        count, sentiment = 0, []
    return outcome, count, sentiment

def audio_analyzer(scene, person1, person2, audio_db):
    User = Query()
    sentiment = []
    counter = 0
    results = audio_db.search((User["label"] == person1) | (User["label"] == person2))
    for i in results:
        blob = TextBlob(i["text"].lower())
        if blob.sentiment.polarity != 0:
            sentiment.append(blob.sentiment.polarity)
        if (i["label"] == person1 and person2 in i["text"]) or i["label"] == (person2 and person1 in i["text"]):
            counter += 1
            
    return counter, sentiment
    
def get_stats(name1, name2, movie, path, vision_db, video_db, audio_db, location_classes, action_classes):
    scene_counter = 0
    shot_counter = 0
    text_counter = 0
    emotions_list = []
    locations_list = []
    action_list = []
    text_emotions = []
    for scene in os.listdir(path):
        scene_return, text_count, text_sentiment = scene_character_checker(name1, name2, scene, vision_db, audio_db)
        if scene_return:
            scene_counter +=1
            text_counter += text_count
            text_emotions.append(text_sentiment)
        for shots in os.listdir(f"{path}/{scene}"):
            shot_return, emotions, locations, action = character_checker(name1, name2, shots, scene, vision_db, video_db)
            if len(emotions) > 0:
                emotions_list.append(emotions)
            if len(locations) > 0:
                if locations != None:
                    locations_list.append(locations)
            if shot_return:
                shot_counter += 1
            if action != None:
                action_list.append(action)
                
    emotions_final = [j for i in emotions_list for j in i]
    emo_hist = get_emo_hist(emotions_final)

    locations_final = [j for i in locations_list for j in i]
    location_hist = get_top3_locations(locations_final, location_classes)

    action_hist = get_top3_actions(action_list, action_classes)

    sentiment_final = [j for i in text_emotions for j in i]
    if len(sentiment_final) > 0:
        average_sentiment = sum(sentiment_final)/len(sentiment_final)
    else:
        average_sentiment = 0
    return scene_counter, shot_counter, text_counter, emo_hist, location_hist, action_hist, average_sentiment

def get_person_location_stats(location, person, vision_db, location_classes):
    
    ### delf features ###
    results = vision_db.search(where('location') == location)
    counter = 0
    for i in results:
        if person in i["faces"]:
            counter += 1
    
    #### places365 features ###
    places_list = []
    User = Query()
    found = vision_db.search(User.faces.any(person))
    for i in found:
        if i["places365"] != None:
            places_list.append(i["places365"])

    places_hist = get_top3_locations(places_list, location_classes)
    
    return counter, places_hist

def get_person_concept_stats(concept, person, audio_db):
    
    ### talk about features ###
    results = audio_db.search(where('label') == person)
    counter = 0
    average_polarity = 0
    polarity_list = []
    emotion_list = []
    for i in results:
        if concept.lower() in i["text"].lower():
            counter += 1
        blob = TextBlob(i["text"].lower())
        if blob.sentiment.polarity != 0:
            polarity_list.append(blob.sentiment.polarity)
        emotion_list.append(i["emotion"][0])

    emotion_hist = get_audio_emo_hist(emotion_list)

    if len(polarity_list) > 0: 
        average_polarity = sum(polarity_list)/len(polarity_list)

    return counter, average_polarity, emotion_hist

def get_emo_hist(lst):
    counter = Counter(lst)
    return counter

def get_top3_actions(lst, action_classes):
    a = [i for i in lst if i != None]
    counter=Counter(a)     
    top = counter.most_common(3)
    return_list = []

    for i in top:
        return_list.append([action_classes[i[0]],i[1]])

    if len(return_list) < 3:
        if len(return_list) == 2:
            return_list.append([0,0])
        elif len(return_list) == 1:
            return_list.append([0,0])
            return_list.append([0,0])
        elif len(return_list) == 0:
            return_list = [[0,0],[0,0],[0,0]]

    return return_list

def get_top3_locations(lst, location_classes):
    a = [i for i in lst if i != None]
    counter=Counter(a)
    top = counter.most_common(3)
    return_list = []

    for i in top:
        loc_index = location_classes.index(i[0])
        return_list.append([loc_index,i[1]])

    if len(return_list) < 3:
        if len(return_list) == 2:
            return_list.append([0,0])
        elif len(return_list) == 1:
            return_list.append([0,0])
            return_list.append([0,0])
        elif len(return_list) == 0:
            return_list = [[0,0],[0,0],[0,0]]

    return return_list

def get_action_classes(loc):
    with open(f"{loc}/video/kinetics_classnames.json", "r") as f:
        kinetics_classnames = json.load(f)

        # Create an id to label name mapping
        kinetics_id_to_classname = {}
        for k, v in kinetics_classnames.items():
            kinetics_id_to_classname[str(k).replace('"', "")] = v
        return kinetics_id_to_classname

def get_location_classes(loc):
    classes = list()
    with open(f"{loc}/vision/places365/labels/categories_places365.txt") as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    return classes

def get_audio_emo_hist(lst):
    counter = Counter(lst)
    return counter