import face_recognition
import os
import json


character_information = "/media/lkunam/My Passport/HLVU/training/movie_knowledge_graphs/honey/images"

known_image = face_recognition.load_image_file(character_information + "/Ashley_1.png")

biden_encoding = face_recognition.face_encodings(known_image)

print(len(biden_encoding))

