import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
import torch
from run_vision_stream import run as run_vision
from preprocessing_vision import shot_segment
from run_audio_stream import audio_stream as run_audio
from run_video_stream import video_stream as run_video
from cluster_combine import combiner
from combine_audio_vision import audio_vision_combiner
import settings as s
from vision import data_creation
import audio.audio_preprocessing.extract_audio as extractor
from video import video_preprocessing as video



#############################  paths and movie-list ###################################

movie_list = [ "Road_To_Bali", "Bagman", "The_Illusionist"]

hlvu_location = s.HLVU_LOCATION_TEST
code_loc = s.DIR_PATH
video_path = f"{hlvu_location}/keyframes/shot_split_video"
audio_path = f"{hlvu_location}/audio"
audio_chunk_path = f"{hlvu_location}/audiochunk"
img_path = f"{hlvu_location}/keyframes/shot_keyf"

#######################################################################################


########################### preprocessing hlvu data set ###############################

#shot_segment(hlvu_location)
#data_creation(hlvu_location, testset = True)

# extract audio from video
#extractor.run_extractor(hlvu_location)

#video.process_data(video_path)

########################################################################################

########################## vision stream ###############################################
"""
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
    run_vision(movie_list, hlvu_location, code_loc, img_path, testset= True)
  except RuntimeError as e:
    #Visible devices must be set before GPUs have been initialized
    #print(e)
    print("GPU not working")

########################################################################################

########################## audio stream ###############################################
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(0.7, 0)


if not os.path.exists(audio_chunk_path):
    os.mkdir(audio_chunk_path)

run_audio(hlvu_location, movie_list, audio_path, code_loc)

########################################################################################

########################## video stream ###############################################

run_video(video_path, code_loc, hlvu_location, code_loc, testset = True)

#######################################################################################
"""

combiner(movie_list, hlvu_location, code_loc, img_path, testset = True)
audio_vision_combiner(movie_list,hlvu_location, code_loc)

