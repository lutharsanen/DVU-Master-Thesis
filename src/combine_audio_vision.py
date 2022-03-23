from tinydb.storages import JSONStorage # added. missing in readme.
from tinydb import TinyDB, Query
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer

from datetime import datetime

serialization1 = SerializationMiddleware(JSONStorage)
serialization1.register_serializer(DateTimeSerializer(), 'TinyDate')
serialization2 = SerializationMiddleware(JSONStorage)
serialization2.register_serializer(DateTimeSerializer(), 'TinyDate')

db_audio = TinyDB('audio_honey.json', storage=serialization1)
db_vision = TinyDB('vision_honey.json', storage=serialization2)

speakers = []
for i in db_audio.all():
    speakers.append(i["label"])
speaker_set = set(speakers)


def most_frequent(List):
    return max(set(List), key = List.count)


for speaker in speaker_set:
    query = Query()

    results = db_audio.search(query.label == speaker)

    face_list = []

    for result in results:
        start = result["start"]
        end = result["end"]
        answer = db_vision.search((query.timestamp > start) & (query.timestamp < end))
        if len(answer) > 0:
            if len(answer[0]["faces"]) > 0:
                for faces in answer[0]["faces"]:
                    face_list.append(faces)

    print(most_frequent(face_list), speaker)