import joblib
from pandasql import sqldf


def p2p_evaluator(df_person, binary_model_loc, clf_model_loc):
    X_test_person = df_person.iloc[:,3:24]
    clf = joblib.load(binary_model_loc)
    y_predicted_person = clf.predict(X_test_person)

    df_person["predicted"] = y_predicted_person
    df_person_relation = sqldf(f"SELECT * FROM df_person WHERE predicted=1", locals())

    X_test_person = df_person_relation.iloc[:,3:24]

    clf = joblib.load(clf_model_loc)
    y_predicted_person_relation = clf.predict(X_test_person)
    y_person_proba = clf.predict_proba(X_test_person)

    return y_predicted_person_relation, y_person_proba, df_person_relation



def p2l_evaluator(df_location, binary_model_loc, clf_model_loc):
    X_test_location = df_location.iloc[:, 3:10]
    clf = joblib.load(binary_model_loc)
    y_predicted_location = clf.predict(X_test_location)

    df_location["predicted"] = y_predicted_location
    df_location_relation = sqldf(f"SELECT * FROM df_location WHERE predicted=1", locals())

    X_test_location = df_location_relation.iloc[:, 3:10]

    clf = joblib.load(clf_model_loc)
    y_predicted_location_relation = clf.predict(X_test_location)
    y_location_proba = clf.predict_proba(X_test_location)
    return y_predicted_location_relation,y_location_proba, df_location_relation
