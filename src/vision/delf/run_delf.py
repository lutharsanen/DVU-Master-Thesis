import os
from tqdm import tqdm
from vision.delf.helper import match_images

def compare(image1, image_path, loc_path, delf):
    for location in os.listdir(loc_path):
        for loc_image in os.listdir(f"{loc_path}/{location}"):
            if match_images(f"{image_path}/{image1}",f"{loc_path}/{location}/{loc_image}", delf):
                #print(f"{image1} is in {location}.")
                return location     
    return "unknown"




