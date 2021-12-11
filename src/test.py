import os
import shutil


#character_information = "/media/lkunam/My Passport/HLVU/training/movie_knowledge_graphs/honey/images"
#
#known_image = face_recognition.load_image_file(character_information + "/Ashley_1.png")
#
#biden_encoding = face_recognition.face_encodings(known_image)
#
#print(len(biden_encoding))

dir = "/media/lkunam/My Passport/HLVU/training/movie.keyframes/shot_keyf/"

for files in os.listdir(dir):
    for file in os.listdir(f"{dir}/{files}"):
        file_name = file[:9]
        dir_path = f"{dir}/{files}/{file_name}"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = f"{dir}/{files}/{file}"
        shutil.move(file_path, dir_path)