import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

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
  # dl = ds_train.map(lambda image: resize_image(resolution, image), num_parallel_calls=tf.data.AUTOTUNE).unbatch()
  dl = ds_train.map(partial(resize_image, resolution), num_parallel_calls=tf.data.AUTOTUNE).unbatch()
  dl = dl.shuffle(400).batch(bs, drop_remainder=True).prefetch(1).repeat()
  return dl

class DisplayCallBack(tf.keras.callbacks.Callback):
    
  def __init__(self):
    
    super(DisplayCallBack, self).__init__()

  def on_batch_begin(self, batch, logs=None):
    
    if (batch + 1) % 5000 == 0 or batch == 0:
      img = self.model.generate_images(batch_size=1)
      img_1 = img[0]/K.max(img[0])
      # img_1 = img[0]*0.5 + 0.5
      img_out = tf.math.abs(img[0] - tf.reduce_mean(img[0]))/(tf.math.reduce_std(img[0]))
      img_out = tf.clip_by_value(img_out, 0, 1.0)

      f = plt.figure(figsize=(12, 24))
      f.add_subplot(1,2, 1)
      plt.imshow(img_1)
      plt.axis('off')
      f.add_subplot(1,2, 2)
      plt.imshow(img_out)
      plt.axis('off')
      plt.show(block=True)

