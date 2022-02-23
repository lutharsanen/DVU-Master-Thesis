import shutil
import os

def process_data(file_path):
    # get name of all movies
    movie_list = [i.partition('-')[0] for i in os.listdir(file_path)]
    movie_list = set(movie_list)
    # create movie path
    for movie in movie_list:
        os.mkdir(f"{file_path}/{movie}")

    for file in os.listdir(file_path):
        if os.path.isfile(f"{file_path}/{file}"):
            movie_name = file.partition("-")[0]
            shutil.move(f"{file_path}/{file}", f"{file_path}/{movie_name}")