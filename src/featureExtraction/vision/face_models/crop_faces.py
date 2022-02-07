from facenet_pytorch import MTCNN
from PIL import Image

def crop_unrecognized_faces(image, unknown_counter,cluster_path):
    for i in range(2):
        mtcnn = MTCNN(image_size=160, margin=10, keep_all=True, min_face_size=20)
        boxes = mtcnn.detect(rgb_img)
        img = Image.open(image)
        rgb_img = img.convert("RGB")
    
    for box in boxes[0]:
        im1 = img.crop(list(box))
        rgba_img = im1.convert("RGB")
        rgba_img.save(f"{cluster_path}/unknown_{unknown_counter}.jpg")
        unknown_counter += 1

    return unknown_counter
