from deepface import DeepFace

backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

person_images = "/media/lkunam/DVU-Challenge/HLVU/movie_knowledge_graph/honey/image_copy/Person/"

test_images = "/media/lkunam/DVU-Challenge/HLVU/keyframes/shot_keyf/honey-11/shot_0005/shot_0005_img_1.jpg"

##face detection and alignment
#detected_face = DeepFace.detectFace(img_path = "img.jpg", detector_backend = backends[4])
#
##face verification
#obj = DeepFace.verify(img1_path = "img1.jpg", img2_path = "img2.jpg", detector_backend = backends[4])

#face recognition
df = DeepFace.find(img_path = "/media/lkunam/DVU-Challenge/HLVU/keyframes/shot_keyf/honey-7/shot_0005/shot_0005_img_2.jpg", db_path = "/media/lkunam/DVU-Challenge/HLVU/movie_knowledge_graph/honey/image_copy/", detector_backend = backends[3], enforce_detection = False)

print(df)
#facial analysis
#demography = DeepFace.analyze(img_path = "img4.jpg", detector_backend = backends[4])