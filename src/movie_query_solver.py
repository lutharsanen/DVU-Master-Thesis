import xml.etree.ElementTree as ET
import xmltodict
import json
import xml.dom.minidom as minidom
from tqdm import tqdm
from pandasql import sqldf
import pandas as pd
import networkx as nx
from relationship_helper.query_helper import get_relations, get_source, count_relations
from classifier import movie_evaluater as me
from collections import Counter
import joblib
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")


def movie_queries(location_file, person_file, movie_list, code_loc, hlvu_location, hlvu_test):
    df_location = pd.read_json(location_file)
    df_person = pd.read_json(person_file)
    #df_concept = pd.read_json(concept_file)

    y_predicted_person, y_person_proba, df_person_relation = me.p2p_evaluator(
        df_person, f"{code_loc}/models/person_binary_clf.sav", f"{code_loc}/models/person_classifier.sav")
    y_predicted_location, y_location_proba, df_location_relation = me.p2l_evaluator(
        df_location, f"{code_loc}/models/location_binary_clf.sav", f"{code_loc}/models/location_classifier.sav")

    # define an empty list
    person_list = []
    location_list = []
    #concept_list = []

    # open file and read the content in a list
    with open(f'{code_loc}/data/people.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            relation = line[:-1]
            # add item to the list
            person_list.append(relation)
            
    # open file and read the content in a list
    with open(f'{code_loc}/data/location.txt', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            relation = line[:-1]
            # add item to the list
            location_list.append(relation)

    relation_pers = [person_list[i] for i in y_predicted_person]
    relation_location = [location_list[i] for i in y_predicted_location]


    df_person_graph = pd.DataFrame(np.column_stack([df_person_relation["person1"] ,df_person_relation["person2"], df_person_relation["movie"], relation_pers]), 
                                columns=['person1', 'person2', 'movie', 'relation'])
    df_location_graph = pd.DataFrame(np.column_stack([df_location_relation["location"] ,df_location_relation["person"], df_location_relation["movie"], relation_location]), 
                                columns=['location', 'person', 'movie', 'relation'])
        
    df_relations = pd.read_excel(f'{hlvu_location}/movie_knowledge_graph/HLVU_Relationships_Definitions.xlsx')
    relation_dict = {}

    for _,row in df_relations.iterrows():
        relation_dict[row["Relation"].lower()] = row["Inverse"].lower()

    person_list_inverse = [relation_dict[i] for i in person_list if i in relation_dict.keys()]
    location_list_inverse = [relation_dict[i] for i in location_list if i in relation_dict.keys()]

    for movie in movie_list:

        print(movie)

        df_movie_person = sqldf(f"SELECT * FROM df_person_graph WHERE movie='{movie}'",locals())
        df_movie_location = sqldf(f"SELECT * FROM df_location_graph WHERE movie='{movie}'", locals())

        G = nx.DiGraph()
        G_alt = nx.DiGraph()

        for idx, row in df_movie_person.iterrows():
            if row["relation"] != "unknown":
                if row["relation"] in list(relation_dict.keys()):
                    G.add_edge(row["person1"],row["person2"],label=row["relation"], weights = y_person_proba[idx])
                    G.edges(data=True)

                    G.add_edge(row["person2"],row["person1"],label=relation_dict[row["relation"]], weights = y_person_proba[idx])
                    G.edges(data=True)
                    
                    G_alt.add_edge(row["person1"],row["person2"],label=row["relation"])
                    G_alt.edges(data=True)


        for idx, row in df_movie_location.iterrows():
            if row["relation"] != "unknown":
                if row["relation"] in list(relation_dict.keys()):
                    G.add_edge(row["location"],row["person"],label=row["relation"], weights = y_location_proba[idx])
                    G.edges(data=True)

                    G.add_edge(row["person"],row["location"],label=relation_dict[row["relation"]], weights = y_location_proba[idx])
                    G.edges(data=True)
                    
                    G_alt.add_edge(row["location"],row["person"],label=row["relation"])
                    G_alt.edges(data=True)

        if not os.path.exists(f"{code_loc}/movie_knowledge_graph"):
            os.mkdir(f"{code_loc}/movie_knowledge_graph")

        nx.write_graphml_lxml(G_alt, f"{code_loc}/movie_knowledge_graph/{movie}.graphml")

        path = f"{hlvu_test}/Queries/Movie-level/{movie}.XMLQueries.xml"

        with open(path) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())

        question = list(data_dict["DeepVideoUnderstandingQueries"].values())[1]

        root = minidom.Document()
        
        xml = root.createElement('DeepVideoUnderstandingResults') 
        xml.setAttribute('movie', movie)
        root.appendChild(xml)

        all_nodes = list(G.nodes())
        for q in tqdm(question):
            query_type = str(q["@question"])
            #query_id = str(q["@id"])
            
            if query_type == "1":
                person1 = q["item"][0]["@source"].lower()
                person2 = q["item"][0]["@target"].lower()
                #print(person1, person2)
                if person1 in all_nodes and person2 in all_nodes:
                    paths = nx.all_simple_paths(G, source = person1, target=person2, cutoff = 7)
                    for idx, combi in enumerate(list(paths)):
                        #print(idx)
                        productChild = root.createElement('DeepVideoUnderstandingTopicResult')
                        productChild.setAttribute('question', q["@question"])
                        productChild.setAttribute('id', q["@id"])
                        productChild.setAttribute('paths', str(idx+1))
                        xml.appendChild(productChild)
                        for x,y in zip(range(0, len(combi)-1), range(1,len(combi))):
                            item = root.createElement('item')
                            edge = G.get_edge_data(combi[x],combi[y])["label"]
                            #print(combi[x], edge,combi[x+1])
                            item.setAttribute('source', combi[x])
                            item.setAttribute('relation', edge)
                            item.setAttribute('target',combi[x+1])
                            productChild.appendChild(item)

            if query_type =="2":
                query2_array = []
                productChild = root.createElement('DeepVideoUnderstandingTopicResult')
                productChild.setAttribute('question', str(q["@question"]))
                productChild.setAttribute('id', str(q["@id"]))
                xml.appendChild(productChild)
                for item in q["item"]:
                    if "@subject" in list(item.keys()):
                        sources = get_source(G, item["@object"].split(":")[1].lower(), item["@predicate"].split(":")[1].lower())
                        if sources != 0:
                            query2_array += sources
                #print(query2_array)
                if len(query2_array) > 0:
                    final = Counter(query2_array).most_common()
                    #print(final, type(final))
                    final_sum = sum([i[1] for i in final])
                    counter = 1
                    for el in final:
                        proba = round(el[1]/final_sum*100)
                        item = root.createElement('item')
                        item.setAttribute('order', str(counter))
                        item.setAttribute('subject', str(el[0]))
                        item.setAttribute('confidence', str(proba))
                        productChild.appendChild(item)
                        counter +=1
                else:
                    item = root.createElement('item')
                    item.setAttribute('order', str(1))
                    item.setAttribute('subject', "none")
                    item.setAttribute('confidence', str(0))
                    productChild.appendChild(item)
    
            if query_type =="3":
                productChild = root.createElement('DeepVideoUnderstandingTopicResult')
                productChild.setAttribute('question', str(q["@question"]))
                productChild.setAttribute('id', str(q["@id"]))
                xml.appendChild(productChild)
                answers = []
                for answer in q["Answers"]["item"]:
                    answers.append(answer["@answer"].lower())
                for item in q["item"]:
                    if "@subject" in list(item.keys()):
                        if "unknown" in item["@predicate"].split(":")[1].lower():
                            if item["@predicate"].split(":")[0].lower() == "relation":
                                #get relation
                                subject = item["@subject"].split(":")[1].lower()
                                target = item["@object"].split(":")[1].lower()
                                #print(subject, target)
                                relation = get_relations(G, subject, target, answers, person_list, person_list_inverse, location_list, location_list_inverse)
                                #print(relation, subject, target)
                                item = root.createElement('item')
                                item.setAttribute('type', "Relation")
                                item.setAttribute('answer', str(relation))
                                productChild.appendChild(item)
                        elif "unknown" in item["@object"].split(":")[1].lower():
                            if item["@object"].split(":")[0].lower() == "entity":
                                #count relations
                                subject = item["@subject"].split(":")[1].lower()
                                relation = item["@predicate"].split(":")[1].lower()
                                if subject in all_nodes:
                                    entity = get_source(G, subject, relation)
                                    item = root.createElement('item')
                                    item.setAttribute('type', "Entity")
                                    if len(entity) > 0:
                                        item.setAttribute('answer', entity[0])
                                    else:
                                        item.setAttribute('answer', "unknown")
                                    productChild.appendChild(item)
                            
                            elif item["@object"].split(":")[0].lower() == "integer":
                                #count relations
                                subject = item["@subject"].split(":")[1].lower()
                                relation = item["@predicate"].split(":")[1].lower()
                                if subject in all_nodes:
                                    count = count_relations(G, subject, relation)
                                    item = root.createElement('item')
                                    item.setAttribute('type', "Integer")
                                    item.setAttribute('answer', str(count))
                                    productChild.appendChild(item)
                            
        xml_str = root.toprettyxml(indent ="\t")
        if not os.path.exists(f"{code_loc}/submissions/movie"):
            if not os.path.exists(f"{code_loc}/submissions"):
                os.mkdir(f"{code_loc}/submissions")
                os.mkdir(f"{code_loc}/submissions/movie")
            else:
                os.mkdir(f"{code_loc}/submissions/movie")
        save_path_file = f"{code_loc}/submissions/movie/movie_{movie}.xml"
        with open(save_path_file, "w") as f:
            f.write(xml_str) 

        