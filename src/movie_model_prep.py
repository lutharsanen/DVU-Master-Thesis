from tinydb import TinyDB,Query,where
from datetime import datetime
from tinydb.storages import JSONStorage
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer

serialization = SerializationMiddleware(JSONStorage)
serialization.register_serializer(DateTimeSerializer(), 'TinyDate')


vision_db = TinyDB('vision_honey.json', storage=serialization)


def character_checker(name1, name2, level):
    User = Query() 
    test = vision_db.search(User.faces.any([name1, name2]) & (where('shots') == level))
    truth_check = [False, False]
    for result in test:
        if name1 in result["faces"]:
            truth_check[0] = True
        if name2 in result["faces"]:
            truth_check[1] = True
        #print(truth_check)
    outcome = all(truth_check)
    return outcome





