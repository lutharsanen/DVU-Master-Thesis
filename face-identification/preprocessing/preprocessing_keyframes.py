import os
import shutil

dir = "/media/lkunam/DVU-Challenge/HLVU/keyframes/shot_keyf/"

for files in os.listdir(dir):
    for file in os.listdir(f"{dir}/{files}"):
        file_name = file[:9]
        dir_path = f"{dir}/{files}/{file_name}"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = f"{dir}/{files}/{file}"
        shutil.move(file_path, dir_path)