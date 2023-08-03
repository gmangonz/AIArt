import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from glob import glob
import os
from Music_Processing import normalize

def _parse_features(example_proto, resolution):
    
    """ Parse TFRecord """

    features = {'image/height': tf.io.FixedLenFeature([], dtype=tf.int64),
                'image/width': tf.io.FixedLenFeature([], dtype=tf.int64),
                'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string)}

    parsed_features = tf.io.parse_single_example(example_proto, features)
    height = parsed_features['image/height']
    width = parsed_features['image/width']

    image = tf.image.decode_image(parsed_features['image/encoded'], channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=(resolution, resolution))
    image = tf.cast(image, tf.float32) / 255
    return image

def get_dataset(ds_directory, resolution, bs=2, block_length=50):

  """ Create tf.data.Dataset """

  # dataset_files = glob(os.path.join(ds_directory, '*'))
  dataset_files = ['/content/drive/MyDrive/DL-CV-ML Projects/All_Data/AIART/dataset2', '/content/drive/MyDrive/DL-CV-ML Projects/All_Data/AIART/dataset1']
  print(dataset_files)
  dataset = tf.data.Dataset.from_tensor_slices(dataset_files)
  dataset = dataset.interleave(tf.data.TFRecordDataset,
                               cycle_length=len(dataset_files),
                               block_length=block_length,
                               num_parallel_calls=tf.data.AUTOTUNE)

  dataset = dataset.ignore_errors()
  dataset = dataset.map(lambda x: _parse_features(x, resolution), num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.shuffle(3_000).batch(bs, drop_remainder=True).repeat()
  return dataset

def resize_image(resolution, image):

  """
  Resize an image, normalize and cast it to tf.float32

  Args:
    resolution: integer for the resolution we want to resize to 

    image: the image that will be resized and normalized
  """

  image = tf.image.resize(image, (resolution, resolution), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image = tf.cast(image/255, tf.float32)
  return image

def make_dataset(resolution, ds_train, bs = 2):
    
  """
  Map the given tf.dataset with a function that has resolution as a stable input while image (2nd input to resize_image) is iterated through the dataset 
  Shuffle, Batch, Prefetch and Repeat the tf.dataset
  
  
  Args:
    resolution: integer for the resolution we want to resize to 

    ds_train: tf.dataset
    
    bs: batch size desired
  
  """

  dl = ds_train.map(partial(resize_image, resolution), num_parallel_calls=tf.data.AUTOTUNE).unbatch()
  dl = dl.shuffle(400).batch(bs, drop_remainder=True).prefetch(1).repeat()
  return dl


class DisplayCallBack(tf.keras.callbacks.Callback):

  def __init__(self):

    super(DisplayCallBack, self).__init__()

  def on_batch_begin(self, batch, logs=None):

    if self.model.train_step_counter % 8000 == 0 or self.model.train_step_counter == 0:
      img = self.model.generate_images(batch_size=1)

      plt.figure(figsize=(5, 5))
      plt.imshow(normalize(img[0], 0, 1))
      plt.show()