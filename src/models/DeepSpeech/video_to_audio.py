import moviepy.editor as mp
import os

# Insert Local Video File Path 

location = "/media/lkunam/DVU-Challenge/DVU-Master-Thesis/src/test-videos"

for file in os.listdir(location):
    print(file)
    clip = mp.VideoFileClip(f"{location}/{file}")
    clip.audio.write_audiofile(f"{file[:-4]}_audio.wav")
