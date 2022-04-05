from vision import data_creation
from vision.visual_preprocessing.shotdetect_f import create_keyframes
import os
import shutil
from tqdm import tqdm

def shot_segment(HLVU_LOCATION):
    hlvu_location = f"{HLVU_LOCATION}/movie.shots" 

    keyframe_path = f"{HLVU_LOCATION}/keyframes"
    os.mkdir(keyframe_path)

    for video in os.listdir(hlvu_location):
        
        dir = f"{hlvu_location}/{video}"
        create_keyframes(dir, keyframe_path,video[:-5])


    movie_list = []
    #print(f"{hlvu_location}/movie.shots")
    for file in tqdm(os.listdir(hlvu_location)):
        shot_frame = file[:-5]
        movie_name = shot_frame.partition("-")
        movie_list.append(movie_name[0])

    all_movies = set(movie_list)

    for movie in all_movies:
        os.mkdir(f"{keyframe_path}/shot_keyf/{movie}")

    dir_path = f"{keyframe_path}/shot_keyf"

    for file in tqdm(os.listdir(dir_path)):
        if file not in all_movies:
            movie_name = file.partition("-")[0]
            file_path = f"{dir_path}/{file}"
            shutil.copytree(file_path, f"{dir_path}/{movie_name}/{file}")
            shutil.rmtree(file_path)
    for movies in tqdm(os.listdir(dir_path)):
        for scenes in os.listdir(f"{dir_path}/{movies}"):
            if not os.path.isfile(f"{dir_path}/{movies}/{scenes}"):
                for shots in os.listdir(f"{dir_path}/{movies}/{scenes}"):
                    file_name = shots[:9]
                    shots_path = f"{dir_path}/{movies}/{scenes}/{file_name}"

                    if not os.path.exists(shots_path):
                        os.makedirs(shots_path)
                    file_path = f"{dir_path}/{movies}/{scenes}/{shots}"
                    shutil.move(file_path, shots_path)