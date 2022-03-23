import settings as s
from vision import start_face_clustering
from tinydb.storages import JSONStorage # added. missing in readme.
from tinydb import TinyDB, Query
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer

from datetime import datetime

serialization = SerializationMiddleware(JSONStorage)
serialization.register_serializer(DateTimeSerializer(), 'TinyDate')

hlvu_location = s.HLVU_LOCATION
dir_path = s.DIR_PATH
img_path = f"{hlvu_location}/keyframes/shot_keyf"



db_vision = TinyDB(f'{dir_path}/database/vision_honey2.json', storage=serialization)

movies = "honey"

cluster_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/clustering"
cluster_df = start_face_clustering(cluster_path)
# then cluster with facenet and replace unknown images by person images
labels = [i.partition("_")[0] for i in cluster_df["name"]]
cluster_df["labels"] = labels
grouping = cluster_df.groupby(["dbscan", "labels"]).size().reset_index(name="freq")
label_class = {}
sub = grouping[grouping.dbscan != -1]
while len(sub) > 0:
    idx = sub['freq'].idxmax()
    maxi = sub.loc[[idx]]
    dbscan, labels, _ = maxi.to_numpy().tolist()[0]
    label_class[dbscan] = labels
    sub = sub[(sub.dbscan != dbscan) & (sub.labels != labels)]

query = Query()
print("comming here")
print(len(db_vision))
for row in db_vision.all():
    print("in the loop")
    image, scene = row["image"],row["scene"]
    replace_face = []
    for face in row["faces"]:
        if "unknown" in face:
            if len(cluster_df.loc[cluster_df['name'] == face]["dbscan"].tolist()) >0:
                lab_class = label_class[cluster_df.loc[cluster_df['name'] == face]["dbscan"].tolist()[0]]
                replace_face.append(lab_class)
            else:
                replace_face.append(face)
        else:
            replace_face.append(face)
    db_vision.update({'faces': replace_face}, (query.image == image) & (query.scene == scene))