import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from Layers import FullyConnectedLayer
from Layers import Conv2dLayer

class MiniBatchSTD(layers.Layer):

  def __init__(self,
               batch_size=None,
               num_new_features=1,
               **kwargs):

    super(MiniBatchSTD, self).__init__(**kwargs)
    self.num_new_features = num_new_features

  def build(self, input_shape):

    _, self.h, self.w, self.c = input_shape # (bs, h, w, c)
    super().build(input_shape)

  def call(self, x):
      
      bs = tf.shape(x)[0]
      group_size = tf.minimum(4, bs)

      x = tf.transpose(x, [0, 3, 1, 2])                       # Change from [NHWC] to [NCHW].
      y = tf.reshape(x, [group_size, -1, self.num_new_features, self.c//self.num_new_features, self.h, self.w]) # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
      y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
      y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
      y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
      y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
      y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
      y = tf.reduce_mean(y, axis=[2])                         # [Mn11]   Split channels into c channel groups.
      y = tf.cast(y, x.dtype)                                 # [Mn11]   Cast back to original data type.
      y = tf.tile(y, [group_size, 1, self.h, self.w])         # [NnHW]   Replicate over group and pixels.
      y = tf.transpose(y, [0, 2, 3, 1])                       # [NHWn]  Move channels to last position.
      x = tf.transpose(x, [0, 2, 3, 1])                       # [NHWn]  Move channels to last position.

      return tf.concat([x, y], axis=-1)


class DiscriminatorBlock(layers.Layer):
  
  """
  Residual connected discriminator block.
  
  In main branch: Perform 2 3x3 convolutions and downsample.
  In residual branch: Downsample and then perform 1x1 convolution.
  
  Add two branches.
  
  """

  def __init__(self,
               out_channels,
               **kwargs):

    super(DiscriminatorBlock, self).__init__(**kwargs)
    self.out_channels = out_channels
    self.resample_kernel = [1, 3, 3, 1]

  def build(self, input_shape):

    self.b, self.h, self.w, self.in_channels = input_shape

    input_layer = layers.Input(shape=(self.h, self.w, self.in_channels), batch_size=self.b)
    x_residual = Conv2dLayer(self.out_channels, kernel_size=1, impl='ref', gpu=True, down=True, resample_kernel=self.resample_kernel)(input_layer)

    x = Conv2dLayer(self.in_channels, kernel_size=3, impl='ref', gpu=True)(input_layer)
    x = layers.LeakyReLU(0.2)(x)
    x = Conv2dLayer(self.out_channels, kernel_size=3, impl='ref', gpu=True, down=True, resample_kernel=self.resample_kernel)(x) 
    x = layers.LeakyReLU(0.2)(x)

    x_add = layers.Add()([x, x_residual]) * tf.sqrt(1/2)
    self.d_block = keras.Model(input_layer, x_add, name='Discriminator_resnet')
    super().build(input_shape)

  def call(self, x):

    return self.d_block(x)


class Discriminator(layers.Layer):

  def __init__(self,
               batch_size=None,
               max_log2_res=10,
               min_num_features=64,
               max_num_features=512,
               **kwargs):
    
    super(Discriminator, self).__init__(**kwargs)

    features = [min(max_num_features, min_num_features * (2 ** i)) for i in range(max_log2_res - 1)]
    num_blocks = len(features) - 1 # -1 because the last 4x4 will be performed outside

    input_image = layers.Input(shape=(2**max_log2_res, 2**max_log2_res, 3), batch_size=batch_size, name='D_input')
    from_rgb = Conv2dLayer(min_num_features, kernel_size=1, name='D_from_rgb_Conv2d')(input_image)
    x = layers.LeakyReLU(0.2)(from_rgb)

    for i in range(num_blocks):
      x = DiscriminatorBlock(features[i+1])(x) # DiscriminatorBlocks will get the input shape with .build()

    x = MiniBatchSTD(batch_size=batch_size, name='MBatchStd')(x)
    x = Conv2dLayer(features[-1] + 1, kernel_size=3, name='Disc_final_Conv2d')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = FullyConnectedLayer(1, bias='zeros', activation=False, name='Final_classifier')(x)

    self.discriminator_blocks = tf.keras.Model(input_image, x, name=f"discriminator_{2**max_log2_res}_x_{2**max_log2_res}")

  def call(self, x):

    assert x.shape[-1] == 3, 'Input x should have 3 channels'    
    return self.discriminator_blocks(x)
  