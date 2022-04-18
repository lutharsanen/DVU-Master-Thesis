import xml.etree.ElementTree as ET
import xmltodict
import json
import xml.dom.minidom as minidom
from tqdm import tqdm
from pandasql import sqldf
import pandas as pd
import networkx as nx
from relationship_helper.query_helper import get_relations, get_source, count_relations
from collections import Counter
import joblib
import os


def movie_queries(location_file, person_file, concept_file, movie_list, code_loc, hlvu_location, hlvu_test):
    df_location = pd.read_json(location_file)
    df_person = pd.read_json(person_file)
    df_concept = pd.read_json(concept_file)
    for movie in movie_list:
        X_test_person = df_person.iloc[:,3:24]
        clf = joblib.load(f"{code_loc}/models/person_classifier.sav")
        y_predicted_person = clf.predict(X_test_person)

        X_test_location = df_location.iloc[:, 3:10]
        clf = joblib.load(f"{code_loc}/models/location_classifier.sav")
        y_predicted_location = clf.predict(X_test_location)



        X_test_concept = df_concept.iloc[:, 3:9]
        if len(X_test_concept) > 0:
            clf = joblib.load(f"{code_loc}/models/concept_classifier.sav")
            y_predicted_person = clf.predict(X_test_concept)

        # define an empty list
        person_list = []
        location_list = []
        concept_list = []

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
                
        # open file and read the content in a list
        #with open(f'{code_loc}/data/concept.txt', 'r') as filehandle:
        #    for line in filehandle:
                # remove linebreak which is the last character of the string
        #        relation = line[:-1]

                # add item to the list
        #        concept_list.append(relation)

        df_person_graph = df_person[['person1', 'person2','movie']]
        df_location_graph = df_location[['location','person','movie']]

        df_person_graph["relation"] = [person_list[i] for i in y_predicted_person]
        df_location_graph["relation"] = [location_list[i] for i in y_predicted_location]


        mysql = lambda q: sqldf(q, globals())
        df_movie_person = mysql(f"SELECT * FROM df_person_graph WHERE movie='{movie}'")
        df_movie_location = mysql(f"SELECT * FROM df_location_graph WHERE movie='{movie}'")

        df_relations = pd.read_excel(f'{hlvu_location}/movie_knowledge_graph/HLVU_Relationships_Definitions.xlsx')
        relation_dict = {}
        for _,row in df_relations.iterrows():
            relation_dict[row["Relation"].lower()] = row["Inverse"].lower()

        G = nx.DiGraph()

        for idx, row in df_movie_person.iterrows():
            if row["relation"] != "unknown":
                G.add_edge(row["person1"],row["person2"],label=row["relation"])
                G.edges(data=True)
                
                G.add_edge(row["person2"],row["person1"],label=relation_dict[row["relation"]])
                G.edges(data=True)
                
        for idx, row in df_movie_location.iterrows():
            if row["relation"] != "unknown":
                G.add_edge(row["location"],row["person"],label=row["relation"])
                G.edges(data=True)
                
                G.add_edge(row["person"],row["location"],label=relation_dict[row["relation"]])
                G.edges(data=True)


        path = f"{hlvu_test}/Queries/Movie-level/{movie}.XMLQueries.xml"

        with open(path) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())

        question = list(data_dict["DeepVideoUnderstandingQueries"].values())[1]

        root = minidom.Document()
        
        xml = root.createElement('DeepVideoUnderstandingResults') 
        xml.setAttribute('movie', 'Bagman')
        root.appendChild(xml)

        for q in tqdm(question):
            query_type = str(q["@question"])
            query_id = str(q["@id"])
            if query_type == "1":
                person1 = q["item"][0]["@source"].lower()
                person2 = q["item"][0]["@target"].lower()
                paths = nx.all_simple_paths(G, source = person1, target=person2)
                for idx, combi in enumerate(list(paths)):
                    productChild = root.createElement('DeepVideoUnderstandingTopicResult')
                    productChild.setAttribute('question', q["@question"])
                    productChild.setAttribute('id', q["@id"])
                    productChild.setAttribute('paths', str(idx+1))
                    xml.appendChild(productChild)
                    for x,y in zip(range(0, len(combi)-1), range(1,len(combi))):
                        item = root.createElement('item')
                        edge = G.get_edge_data(combi[x],combi[y])["label"]
                        item.setAttribute('source', combi[x])
                        item.setAttribute('relation', edge)
                    item.setAttribute('target',combi[-1])
                    productChild.appendChild(item)
            if query_type =="2":
                query2_array = []
                productChild = root.createElement('DeepVideoUnderstandingTopicResult')
                productChild.setAttribute('question', q["@question"])
                productChild.setAttribute('id', q["@id"])
                xml.appendChild(productChild)
                for item in q["item"]:
                    if "@subject" in list(item.keys()):
                        sources = get_source(G, item["@object"].split(":")[1].lower(), item["@predicate"])
                        if sources != 0:
                            query2_array += sources
                print(query2_array)
                if len(query2_array) > 0:
                    final = Counter(query2_array).most_common()
                    final_sum = sum(list(final.values()))
                    counter = 1
                    for key, values in final.items():
                        proba = round(values/final_sum*100)
                        item = root.createElement('item')
                        item.setAttribute('order', str(counter))
                        item.setAttribute('subject', key)
                        item.setAttribute('confidence', str(proba))
                        productChild.appendChild(item)
            if query_type =="3":
                for item in q["item"]:
                    productChild = root.createElement('DeepVideoUnderstandingTopicResult')
                    productChild.setAttribute('question', q["@question"])
                    productChild.setAttribute('id', q["@id"])
                    xml.appendChild(productChild)
                    if "@subject" in list(item.keys()):
                        if "unknown" in item["@predicate"].split(":")[1].lower():
                            #get relation
                            subject = item["@subject"].split(":")[1]
                            target = item["@object"].split(":")[1]
                            relation = get_relations(G, subject, target)
                            item = root.createElement('item')
                            item.setAttribute('type', "Relation")
                            item.setAttribute('answer', relation)
                            productChild.appendChild(item)
                        elif "unknown" in item["@object"].split(":")[1].lower():
                            #count relations
                            subject = item["@subject"].split(":")[1]
                            relation = item["@predicate"].split(":")[1]
                            count = count_relations(G, subject, relation)
                            item = root.createElement('item')
                            item.setAttribute('type', "Integer")
                            item.setAttribute('answer', count)
                            productChild.appendChild(item)

        xml_str = root.toprettyxml(indent ="\t")
        if not os.path.exists(f"{code_loc}/submissions/movie"):
            os.mkdir(f"{code_loc}/submissions/movie")
        save_path_file = f"{code_loc}/submissions/movie/movie_{movie}.xml"
        with open(save_path_file, "w") as f:
            f.write(xml_str) 

        