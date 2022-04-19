from dataclasses import make_dataclass
import pandas as pd
import joblib
import settings
from pandasql import sqldf


def test():
    movie_list = [ "Manos", "Bagman", "Road_To_Bali", "The_Illusionist"]

    code_loc = settings.DIR_PATH
    location_file = f"{code_loc}/data/people2location_test.json"
    person_file = f"{code_loc}/data/people2people_test.json" 
    concept_file =  f"{code_loc}/data/people2concept_test.json"

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

        df_movie_person = sqldf(f"SELECT * FROM df_person_graph WHERE movie='{movie}'",locals())
        df_movie_location = sqldf(f"SELECT * FROM df_location_graph WHERE movie='{movie}'", locals())

        print(df_movie_location)
        print(df_movie_person)



test()