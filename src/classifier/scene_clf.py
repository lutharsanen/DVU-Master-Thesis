import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
import joblib

def interaction_classifier(training_set, model_location, dir_path):
    df = pd.read_json(training_set)
    y_cleaned = [x.strip(" ").lower() if x is not None else "unknown" for x in df["interaction"]]
    relation_list = list(set(y_cleaned))

    y = [relation_list.index(i) for i in y_cleaned]
    X = df.drop(columns=['person1', 'person2', 'interaction'])

    with open(f'{dir_path}/data/interactions.txt', 'w') as filehandle:
        for relation in relation_list:
            filehandle.write('%s\n' % relation)

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    clf = RandomForestClassifier(max_depth=16, random_state=0)
    clf.fit(X_res, y_res)
    joblib.dump(clf, model_location)