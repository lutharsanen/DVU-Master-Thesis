import moviepy.editor as mp
import os
import settings as s

# Insert Local Video File Path 


def run_extractor(hlvu_location):
    movie_list = []
    for file in os.listdir(f"{hlvu_location}/movie.shots"):
        shot_frame = file[:-5]
        movie_name = shot_frame.partition("-")
        movie_list.append(movie_name[0])
    
    os.mkdir(f"{hlvu_location}/audio")
    for i in set(movie_list):
        #print(i)
        os.mkdir(f"{hlvu_location}/audio/{i}")
    
    for file in os.listdir(f"{hlvu_location}/movie.shots"):
        shot_frame = file[:-5]
        movie_name = shot_frame.partition("-")
        movie_name = movie_name[0]
        clip = mp.VideoFileClip(f"{hlvu_location}/movie.shots/{file}")
        clip.audio.write_audiofile(f"{hlvu_location}/audio/{movie_name}/{shot_frame}.wav")

