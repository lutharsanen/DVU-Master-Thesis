from deepface import DeepFace

backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]

#img_path = "/home/user/kunam/HLVU"
img_path = "/media/lkunam/DVU-Challenge/HLVU"
movies = "honey"


df = DeepFace.find(img_path = f"/media/lkunam/DVU-Challenge/thesis-notebooks/deepface_images/fa/img1.jpg", enforce_detection = False, db_path = f"{img_path}/movie_knowledge_graph/{movies}/image/Person/", detector_backend = backends[4])