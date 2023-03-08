import tensorflow as tf
from tensorflow.keras import layers

class Zoom(layers.Layer):

  def __init__(self):

    super(Zoom, self).__init__()
    self.trainable=False

  def call(self, input):

    x, mag = input
    x_t = tf.transpose(x[0], [2, 0, 1]) # (1, H, W, C) -> (C, H, W)
    x_t_zoom = tf.keras.preprocessing.image.random_zoom(x_t, zoom_range=(mag, mag))
    return tf.transpose(x_t_zoom, [1, 2, 0])[None, ...] # (C, H, W) -> (1, H, W, C)

class Translation(layers.Layer): ############## MAKE CHANGES to match zoom_range=(mag, mag)

  def __init__(self):

    super(Translation, self).__init__()
    self.trainable=False

  def call(self, input):

    x, mag = input
    x_t = tf.transpose(x[0], [2, 0, 1]) # (1, H, W, C) -> (C, H, W)
    x_t_shift = tf.keras.preprocessing.image.random_shift(x_t, mag, mag)
    return tf.transpose(x_t_shift, [1, 2, 0])[None, ...] # (C, H, W) -> (1, H, W, C)
    # factor = (tf.cast(tf.math.minimum(mag, 0.95), tf.float64), tf.cast(tf.math.minimum((mag + 0.2*mag), 1), tf.float64))
    # return tf.keras.layers.RandomTranslation(height_factor=factor, width_factor=factor)(x)

class Rotation(layers.Layer): ############## MAKE CHANGES to match zoom_range=(mag, mag)

  def __init__(self):

    super(Rotation, self).__init__()
    self.trainable=False

  def call(self, input):

    x, mag = input

    augmentation_values = tf.random.uniform(shape=(1,), minval=0.0, maxval=1.0) # Generate random number
    pos_or_neg = tf.where(tf.math.less(augmentation_values, mag) , -1, 1).numpy()[0]
    if pos_or_neg == 1:
      factor = (tf.cast(tf.math.minimum(mag, 0.95), tf.float64), tf.cast(tf.math.minimum((mag + 0.2*mag), 1), tf.float64))
    else:
      factor = (tf.cast(tf.math.maximum(-(mag + 0.2*mag), -1), tf.float64), tf.cast(tf.math.maximum(-mag, -0.95), tf.float64))

    x_t = tf.transpose(x[0], [2, 0, 1]) # (1, H, W, C) -> (C, H, W)
    x_t_rotation = tf.keras.preprocessing.image.random_rotation(x_t, factor)
    return tf.transpose(x_t_rotation, [1, 2, 0])[None, ...] # (C, H, W) -> (1, H, W, C)
    return tf.keras.layers.RandomRotation(factor=factor)(x)

class NetworkBending(layers.Layer):

  """
  Performs Network Bending:
      Check the name of the layer.
      If it matches with the specified name, then transform the given input with the desired transformation.
  
  Transformations include: Zoom, Translation, Rotation.
  
  """  
  def __init__(self, 
               layer_name,
               **kwargs):

    super(NetworkBending, self).__init__(**kwargs)
    self.layer_name = layer_name # Does not correspond to THIS layer's name but rather Generator_Block's name
    self.trainable = False

  def call(self, inputs): # inputs: (x, dict('layer_name': [transformation, magnitude]))

    input, transformation_dict = inputs
    for name, (transformation, magnitude) in transformation_dict.items():
      if name == self.layer_name:
        input = transformation([input, magnitude])

    return input