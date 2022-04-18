import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
import joblib


def people_classifier(training_set, model_location, dir_path):
    p2p = pd.read_json(training_set)
    relation_lst = list(set([x.strip(" ").lower() if x is not None else "unknown" for x in p2p["relation"]]))
    with open(f'{dir_path}/data/people.txt', 'w') as filehandle:
        for relation in relation_lst:
            filehandle.write('%s\n' % relation)
    y_train = [relation_lst.index(x.strip(" ").lower()) if x is not None else relation_lst.index("unknown") for x in p2p["relation"]]
    X_train = p2p.iloc[:, 2:23]
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    clf = RandomForestClassifier(max_depth=11, random_state=0)
    clf.fit(X_res, y_res)
    joblib.dump(clf, model_location)


def location_classifier(training_set, model_location, dir_path):
    p2l = pd.read_json(training_set)
    relation_lst = list(set([x.strip(" ").lower() if x is not None else "unknown" for x in p2l["relation"]]))
    with open(f'{dir_path}/data/location.txt', 'w') as filehandle:
        for relation in relation_lst:
            filehandle.write('%s\n' % relation)
    y_train = [relation_lst.index(x.strip(" ").lower()) if x is not None else relation_lst.index("unknown") for x in p2l["relation"]]
    X_train = p2l.iloc[:, 2:9]
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    clf = RandomForestClassifier(max_depth=11, random_state=0)
    clf.fit(X_res, y_res)
    joblib.dump(clf, model_location)


def concept_classifier(training_set, model_location, dir_path):
    p2c = pd.read_json(training_set)
    relation_lst = list(set([x.strip(" ").lower() if x is not None else "unknown" for x in p2c["relation"]]))
    with open(f'{dir_path}/data/concept.txt', 'w') as filehandle:
        for relation in relation_lst:
            filehandle.write('%s\n' % relation)
    y_train = [relation_lst.index(x.strip(" ").lower()) if x is not None else relation_lst.index("unknown") for x in p2c["relation"]]
    X_train = p2c.iloc[:, 2:8]
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    clf = RandomForestClassifier(max_depth=12, random_state=0)
    clf.fit(X_res, y_res)
    joblib.dump(clf, model_location)
