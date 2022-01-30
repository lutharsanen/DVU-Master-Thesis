import settings
import os

hlvu_location = settings.HLVU_LOCATION
movies = "honey"
model_loc = f"{hlvu_location}/movie_knowledge_graph/{movies}/image/representations_vgg_face.pkl"

if os.path.isfile(model_loc):
    os.remove(model_loc)
    print("model deleted")
