import tensorflow as tf
from PIL import Image, ImageOps

import numpy as np
import tensorflow_hub as hub

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform

from PIL import Image
import imagehash

def check_similarity(image1,image2):
  hash0 = imagehash.average_hash(Image.open(image1)) 
  hash1 = imagehash.average_hash(Image.open(image2)) 
  cutoff = 30  # maximum bits that could be different between the hashes. 

  if hash0 - hash1 < cutoff:
    return True
  else:
    return False


def download_and_resize(path, new_width=256, new_height=256):
  image = Image.open(path)
  if np.array(image).shape[2] != 3:
    image = image.convert('RGB')
  image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
  return image

def run_delf(image):
    delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']
    np_image = np.array(image)
    float_image = tf.image.convert_image_dtype(np_image, tf.float32)

    return delf(
      image=float_image,
      score_threshold=tf.constant(100.0),
      image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
      max_feature_num=tf.constant(1000))

#@title TensorFlow is not needed for this post-processing and visualization
def match_images(image1, image2):
  if check_similarity(image1, image2):
    image1 = download_and_resize(image1)
    image2 = download_and_resize(image2)
    result1 = run_delf(image1)
    result2 = run_delf(image2)
    distance_threshold = 0.8

     # Read features.
    num_features_1 = result1['locations'].shape[0]
    #print("Loaded image 1's %d features" % num_features_1)

    num_features_2 = result2['locations'].shape[0]
    #print("Loaded image 2's %d features" % num_features_2)

     # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(result1['descriptors'])
    _, indices = d1_tree.query(
        result2['descriptors'],
        distance_upper_bound=distance_threshold)

     # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        result2['locations'][i,]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        result1['locations'][indices[i],]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

    if (len(locations_2_to_use) and (len(locations_1_to_use))) > 3:
        # Perform geometric verification using RANSAC.
        _, inliers = ransac(
            (locations_1_to_use, locations_2_to_use),
            AffineTransform,
            min_samples=3,
            residual_threshold=20,
            max_trials=1000)

        if type(inliers) != np.ndarray:
          print(inliers)
          return False

        print('Found %d inliers' % sum(inliers))
        if sum(inliers) > 20:
          return True
        else:
          return False
    else:
      return False

  else:
    return False

