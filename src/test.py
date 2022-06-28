from dataclasses import make_dataclass
import pandas as pd
import joblib
import settings
from pandasql import sqldf
from classifier.movie_clf import *
import settings as s
from movie_query_solver import movie_queries

hlvu_location = s.HLVU_LOCATION_TEST
hlvu_training = s.HLVU_LOCATION
code_loc = s.DIR_PATH
video_path = f"{hlvu_location}/keyframes/shot_split_video"
audio_path = f"{hlvu_location}/audio"
audio_chunk_path = f"{hlvu_location}/audiochunk"
img_path = f"{hlvu_location}/keyframes/shot_keyf"
movie_list = [ "Calloused_Hands", "ChainedforLife", "Liberty_Kid", "like_me", "little_rock", "losing_ground"]


people_classifier(f"{code_loc}/data/people2people.json", f"{code_loc}/models/person_binary_clf.sav", f"{code_loc}/models/person_classifier.sav", code_loc)
location_classifier(f"{code_loc}/data/people2location.json", f"{code_loc}/models/location_binary_clf.sav", f"{code_loc}/models/location_classifier.sav", code_loc)

movie_queries(
  f"{code_loc}/data/people2location_test.json", 
  f"{code_loc}/data/people2people_test.json",
  movie_list, code_loc, hlvu_training , hlvu_location)