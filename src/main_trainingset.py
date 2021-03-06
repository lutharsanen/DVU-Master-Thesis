import tensorflow as tf
#import torch
from classifier.scene_clf import interaction_classifier
#from run_vision_stream import run as run_vision
#from preprocessing_vision import shot_segment
#from run_audio_stream import audio_stream as run_audio
#from run_video_stream import video_stream as run_video
#from cluster_combine import combiner
#from combine_audio_vision import audio_vision_combiner
#import audio.audio_preprocessing.extract_audio as extractor
#from vision import training
#from vision import data_creation
#from video import video_preprocessing as video
import settings as s
import os
import movie_model_prep as m
from classifier.movie_clf import *
from movie_query_solver import movie_queries
from scene_model_prep import scene_data_creation

movie_list = ["shooters", "The_Big_Something", "time_expired", "Valkaama", "Huckleberry_Finn", "spiritual_contact", "honey", "sophie", "Nuclear_Family", "SuperHero","Bagman", "Manos", "The_Illusionist", "Road_To_Bali"]

hlvu_location = s.HLVU_LOCATION
code_loc = s.DIR_PATH
video_path = f"{hlvu_location}/keyframes/shot_split_video"
audio_path = f"{hlvu_location}/audio"
audio_chunk_path = f"{hlvu_location}/audiochunk"
img_path = f"{hlvu_location}/keyframes/shot_keyf"

"""

#shot_segment(hlvu_location)
data_creation(hlvu_location)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4500)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    torch.cuda.set_device(0)
    run_vision(movie_list, hlvu_location, code_loc, img_path)
  except RuntimeError as e:
    #Visible devices must be set before GPUs have been initialized
    print(e)

torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(0.4, 0)
# extract audio from video
#extractor.run_extractor()

if not os.path.exists(audio_chunk_path):
  os.mkdir(audio_chunk_path)
run_audio(hlvu_location, movie_list, audio_path, code_loc)

video.process_data(video_path)
run_video(video_path, code_loc, hlvu_location, code_loc)

combiner(movie_list, hlvu_location, code_loc, img_path)
audio_vision_combiner(movie_list,hlvu_location, code_loc)


m.create_dataframe(movie_list, code_loc , hlvu_location)
"""
############################## movie-level ########################################
if not os.path.exists(f"{code_loc}/models"):
  os.mkdir(f"{code_loc}/models")
people_classifier(f"{code_loc}/data/people2people.json", f"{code_loc}/models/person_binary_clf.sav", f"{code_loc}/models/person_classifier.sav", code_loc)
location_classifier(f"{code_loc}/data/people2location.json", f"{code_loc}/models/location_binary_clf.sav", f"{code_loc}/models/location_classifier.sav", code_loc)
#concept_classifier(f"{code_loc}/data/people2concept.json", f"{code_loc}/models/concept_classifier.sav", code_loc)


############################# scene-level #########################################

#scene_data_creation(movie_list, code_loc, hlvu_location)
#interaction_classifier(f"{code_loc}/data/df_interaction.json", f"{code_loc}/models/interaction_classifier.sav", code_loc)
