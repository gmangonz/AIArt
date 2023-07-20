import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

"""
All inputs will be between 0-1

"""


class Zoom(layers.Layer):

  def __init__(self):

    super(Zoom, self).__init__()
    self.trainable=False

  def call(self, x, frame_value):

    x_t = tf.transpose(x[0], [2, 0, 1]) # (1, H, W, C) -> (C, H, W)
    x_t_zoom = tf.keras.preprocessing.image.random_zoom(x_t, zoom_range=(frame_value, frame_value))
    return tf.transpose(x_t_zoom, [1, 2, 0])[None, ...] # (C, H, W) -> (1, H, W, C)


class GaussianBlur(layers.Layer):

  def __init__(self,
                kernel_size,
                max_sigma = 4,
                **kwargs):
    super(GaussianBlur, self).__init__(**kwargs)

    self.kernel_size = kernel_size
    self.max_sigma = tf.cast(max_sigma, tf.float32)

  def gaussian_kernel(self, size=3, sigma=1):

    x_range = tf.range(-(size-1)//2, (size-1)//2 + 1, 1)
    y_range = tf.range((size-1)//2, -(size-1)//2 - 1, -1)

    xs, ys = tf.meshgrid(x_range, y_range)
    kernel = -(xs**2 + ys**2)
    kernel = kernel / tf.cast((2*(sigma**2)), kernel.dtype)
    kernel = tf.exp(kernel)
    kernel = tf.cast(kernel, tf.float32) / (tf.cast(2*np.pi, tf.float32) * tf.cast(sigma**2, tf.float32))
    return kernel / tf.reduce_sum(kernel)

  def blur_image(self, img, sigma):

    kernel = self.gaussian_kernel(self.kernel_size, sigma)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)

    r, g, b = tf.split(img, [1,1,1], axis=-1)
    r_blur = tf.nn.conv2d(r, kernel, [1,1,1,1], 'SAME', name='r_blur')
    g_blur = tf.nn.conv2d(g, kernel, [1,1,1,1], 'SAME', name='g_blur')
    b_blur = tf.nn.conv2d(b, kernel, [1,1,1,1], 'SAME', name='b_blur')

    blur_image = tf.concat([r_blur, g_blur, b_blur], axis=-1)
    blur_image = tf.cast(blur_image, img.dtype)
    return blur_image

  def call(self, x, frame_value):

    sigma = self.max_sigma * frame_value
    sigma = tf.maximum(1.0, sigma)
    return self.blur_image(x, sigma)


class Translation(layers.Layer):

  def __init__(self, max_translation=15, **kwargs):

    super(Translation, self).__init__(**kwargs)
    self.trainable=False
    self.max_translation = tf.cast(max_translation, tf.float32)

  def call(self, x, frame_value):

    mag = tf.cast(self.max_translation * frame_value, tf.int32)
    x_t = tf.transpose(x[0], [2, 0, 1]) # (1, H, W, C) -> (C, H, W)
    x_t_translate = tf.keras.preprocessing.image.apply_affine_transform(x_t, tx=-mag)
    return tf.transpose(x_t_translate, [1, 2, 0])[None, ...] # (C, H, W) -> (1, H, W, C)


class Rotation(layers.Layer):

  def __init__(self, **kwargs):

    super(Rotation, self).__init__(**kwargs)
    self.trainable = False

  def call(self, x, frame_value):

    degree_angle = frame_value * 20
    imgT = tf.transpose(x[0], [2, 0, 1]) # (1, H, W, C) -> (C, H, W)
    img_Rotate = tf.keras.preprocessing.image.apply_affine_transform(imgT, theta=degree_angle, row_axis=1, col_axis=2, channel_axis=0)
    img_Rotate = tf.transpose(img_Rotate, [1, 2, 0])[None, ...] # (C, H, W) -> (1, H, W, C)
    return img_Rotate


class Shear(layers.Layer):

  def __init__(self, max_shear=50, **kwargs):

    super(Shear, self).__init__(**kwargs)
    self.trainable=False
    self.max_shear=tf.cast(max_shear, tf.float32)

  def call(self, x, frame_value):

    shear_angle = self.max_shear * frame_value
    imgT = tf.transpose(x[0], [2, 0, 1]) # (1, H, W, C) -> (C, H, W)
    img_shear = tf.keras.preprocessing.image.apply_affine_transform(imgT, shear=shear_angle, row_axis=1, col_axis=2, channel_axis=0)
    img_shear = tf.transpose(img_shear, [1, 2, 0])[None, ...] # (C, H, W) -> (1, H, W, C)
    return img_shear


class Blend(layers.Layer):

  def __init__(self, **kwargs):
    super(Blend, self).__init__(**kwargs)
    self.trainable=False

  def call(self, x, frame_value):

    # When factor is 1, it output a white image, else it will be the image blended/faded
    return tfa.image.blend(x, tf.ones(shape=tf.shape(x)), frame_value)


class Hue(layers.Layer):

  def __init__(self, **kwargs):
    super(Hue, self).__init__(**kwargs)
    self.trainable=False

  def call(self, x, frame_value):

    # delta [-1, 1]
    return tf.image.adjust_hue(x, frame_value)


class SwirlEffect(layers.Layer):

  def __init__(self, **kwargs):

    super(SwirlEffect, self).__init__(**kwargs)
    self.trainable=False

  def make_y_displacement(self):

    x_range = tf.range(-1, 1, 1/256)
    _, ys = tf.meshgrid(x_range, x_range)
    y_flow = tf.math.cos(ys*np.pi)
    return y_flow

  def make_x_displacement(self, coeff=0.4, log_scale=2):

    """
    coeff: larger = more number of 'ripples', smaller = less

    log_scale: larger = the sharper the waves, smaller = smoother

    """
    range = 2 * np.pi * 10
    step = range / 512
    x_range = tf.range(-np.pi*10, np.pi*10, step)

    X, Y = tf.meshgrid(x_range, x_range)
    wave = tf.math.sqrt(X ** 2 + Y ** 2)
    wave = tf.math.sin(coeff * wave) * tf.math.log(log_scale * wave)
    return wave

  def call(self, x, frame_value, coeff=0.4, log_scale=2):

    # Factor of range 0-40 is good
    y_disp = self.make_y_displacement()
    x_disp = self.make_x_displacement(coeff, log_scale)
    displacement = tf.stack([y_disp, x_disp], axis=-1)
    return tfa.image.dense_image_warp(x, frame_value*displacement[None, ...])


class DisplacementMap(layers.Layer):

  def __init__(self, **kwargs):

    super(DisplacementMap, self).__init__(**kwargs)
    self.trainable=False

  def call(self, x, flow):

    return tfa.image.dense_image_warp(x, flow)


class NetworkBending(layers.Layer):

  """
  Performs Network Bending:
      Check the name of the layer.
      If it matches with the specified name, then transform the given input with the desired transformation.
  """
  def __init__(self,
               layer_name,
               **kwargs):

    super(NetworkBending, self).__init__(**kwargs)
    self.layer_name = layer_name # Does not correspond to THIS layer's name but rather Generator_Block's name
    self.trainable = False

  def call(self, inputs): # inputs: (x, dict('layer_name': [transformation, magnitude]))

    input, transformation_dict = inputs
    # for name, (transformation, magnitude) in transformation_dict.items():
    #   if name == self.layer_name:
    #     input = transformation([input, magnitude])

    for name, transformation_func in transformation_dict.items():
      if name == self.layer_name:
        input = transformation_func(input)
    return input