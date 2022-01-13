import tensorflow as tf
from PIL import Image, ImageOps

import numpy as np
import tensorflow_hub as hub


def download_and_resize(name, url, new_width=256, new_height=256):
  path = tf.keras.utils.get_file(url.split('/')[-1], url)
  image = Image.open(path)
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