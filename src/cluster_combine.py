import settings as s
from vision import start_face_clustering
from tinydb.storages import JSONStorage # added. missing in readme.
from tinydb import TinyDB, Query
from tinydb_serialization import SerializationMiddleware
from tinydb_serialization.serializers import DateTimeSerializer

from datetime import datetime


#hlvu_location = s.HLVU_LOCATION
#dir_path = s.DIR_PATH
#img_path = f"{hlvu_location}/keyframes/shot_keyf"


"""
movie_list = ["shooters", 
              "The_Big_Something", 
              "time_expired", 
              "Valkaama", 
              "Huckleberry_Finn", 
              "spiritual_contact", 
              "honey", 
              "sophie", 
              "Nuclear_Family", 
              "SuperHero"
             ]
"""

def combiner(movie_list, hlvu_location, dir_path, img_path, testset = False):
    serialization = SerializationMiddleware(JSONStorage)
    serialization.register_serializer(DateTimeSerializer(), 'TinyDate')

    for movies in movie_list:
        db_vision = TinyDB(f'{dir_path}/database/vision_{movies}.json', storage=serialization)
   
        if testset == True:
            cluster_path = f"{hlvu_location}/Queries/movie_knowledge_graph/{movies}/clustering"
        else:
            cluster_path = f"{hlvu_location}/movie_knowledge_graph/{movies}/clustering"
        #print(cluster_path)
        cluster_df = start_face_clustering(cluster_path)
        #print(type(cluster_df))
        #if type(cluster_df) ==  int:
        #    print(cluster_df)
        if type(cluster_df) != int:
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
            for row in db_vision.all():
                image, scene = row["image"],row["scene"]
                replace_face = []
                for face in row["faces"]:
                    if "unknown" in face:
                        if len(cluster_df.loc[cluster_df['name'] == face]["dbscan"].tolist()) >0:
                            cluster_label = cluster_df.loc[cluster_df['name'] == face]["dbscan"].tolist()[0]
                            if cluster_label in list(label_class.keys()):
                                lab_class = label_class[cluster_label]
                                replace_face.append(lab_class)
                            else:
                                replace_face.append(face)
                        else:
                            replace_face.append(face)
                    else:
                        replace_face.append(face)
                db_vision.update({'faces': replace_face}, (query.image == image) & (query.scene == scene))
