import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
from dnnlib.ops.upfirdn_2d import upsample_conv_2d, conv_downsample_2d, upsample_2d

class ModulatedConv(tf.keras.layers.Layer):

  def __init__(self, 
               filters, 
               kernel_size=3, 
               strides=1,
               padding='valid',
               dilation_rate=1,
               kernel_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               demod=True,
               gain=1.0, fused_modconv=True, resample_kernel=None, lr_multiplier=1.0, up=False, down=False, impl='ref', gpu=True,
               **kwargs):
        
      super(ModulatedConv, self).__init__(**kwargs)
      self.filters = filters
      self.kernel_size = kernel_size
      self.demod = demod
      self.gain = gain
      self.fused_modconv = fused_modconv
      self.resample_kernel = resample_kernel
      self.lr_multiplier = lr_multiplier
      self.up = up
      self.down = down
      self.impl = impl
      self.gpu = gpu

  def build(self, input_shapes): # [BS, H, W, C]

      assert input_shapes[0][-1] == input_shapes[1][-1] and input_shapes[0][-1] is not None # Check inputs have same number of channels
      self.input_dim = input_shapes[0][-1]

      kernel_shape = [self.kernel_size, self.kernel_size, self.input_dim, self.filters] # w: [kh, kw, in_channels, out_channels]

      fanin = tf.cast(tf.math.reduce_prod( input_shapes[0][1:] ), tf.float32) # H*W*C of x
      self.weight_gain = (self.gain * self.lr_multiplier) / fanin
      self.w = self.add_weight(shape=kernel_shape, initializer=tf.initializers.RandomNormal( mean=0.0, stddev=1.0 / self.lr_multiplier ), trainable=True, name="mod_conv_kernel")

      super().build(input_shapes)
  
  def call(self, inputs):

      x, styles = inputs # x: (bs, h, w in_channels) styles: (bs, in_channels)
      assert styles.shape[0] == x.shape[0], 'x and styles should have same batch size'
      assert styles.shape[-1] == x.shape[-1] == self.input_dim, 'x and styles should have same input dimension'

      batch_size, height, width, _ = x.shape

      w = K.expand_dims( self.w * tf.cast(self.weight_gain, self.w.dtype), axis = 0 ) # (kh, kw, in_channels, out_channels) -> (1, kh, kw, in_channels, out_channels)
      w *= tf.cast( (styles + 1)[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype ) # reshapes styles to (bs, 1, 1, in_channels, 1) and output for w is (bs, kh, kw, in_channels, out_channels)
      
      if self.demod:
        dcoefs = tf.math.rsqrt(tf.reduce_sum(tf.square(w), axis=[1,2,3]) + 1e-8) # (bs, out_channels)
        w = w * tf.reshape(dcoefs, [-1, 1, 1, 1, self.filters]) # (bs, kh, kw, in_channels, out_channels)
      
      if self.fused_modconv:
        
        x = tf.transpose(x, [1, 2, 0, 3]) # (bs, h, w, in_channels) -> (h, w, bs, in_channels)
        x = tf.reshape(x, [1, height, width, -1]) #  (1, h, w, bs*in_channels)

        w = tf.transpose(w, [1, 2, 3, 0, 4]) # (bs, kh, kw, in_channels, out_channels) -> (kh, kw, in_channels, bs, out_channels)
        w = tf.reshape(w, [w.shape[0], w.shape[1], w.shape[2], -1]) # (kh, kw, in_channels, bs, out_channels) - > (kh, kw, in_channels, bs*out_channels)

      else:
        x = x * tf.cast(styles[:, np.newaxis, np.newaxis, :], x.dtype) # (bs, h, w, in_channels) * (bs, 1, 1, in_channels)

      # Convolution with optional up/downsampling.
      if self.up:
          x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NHWC', k=self.resample_kernel, impl=self.impl)
      elif self.down:
          x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NHWC', k=self.resample_kernel, impl=self.impl)
      else:
          x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NHWC', strides=[1,1,1,1], padding='SAME') # (1, h, w, bs*in_channels) * (kh, kw, in_channels, bs*out_channels) -> (1, h, w, bs*out_channels)

      if self.fused_modconv:
          x = tf.transpose(x, [0, 3, 1, 2]) # (1, h', w', bs*out_channels) -> (1, bs*out_channels, h', w')
          x = tf.reshape(x, [-1, self.filters, x.shape[2], x.shape[3]]) # (1, bs*out_channels, h', w') -> (bs, out_channels, h', w')
          x = tf.transpose(x, [0, 2, 3, 1]) # (bs, out_channels, h, w) -> (bs, h, w, out_channels)

      elif self.demod:
          x = x * tf.cast(dcoefs[:, np.newaxis, np.newaxis, :], x.dtype) # (bs, h, w, out_c) * (bs, 1, 1, out_c)

      return x

  def compute_output_shape(self, input_shapes):

      x_shape = input_shapes[0]
      bs, h, w, _ = x_shape.shape

      return (bs, h, w,) + (self.filters,)

class FullyConnectedLayer(tf.keras.layers.Layer):

  def __init__(self, 
               units, 
               bias='zeros', # 'zeros', 'ones'
               activation=True,
               lr_multiplier=1.0, 
               gain=2.0,
               **kwargs):

    super(FullyConnectedLayer, self).__init__(**kwargs)

    self.units = units
    self.lr_multiplier = lr_multiplier
    self.bias_init = bias
    self.gain = gain
    self.activation = activation

  def build(self, input_shape): # (None, in_channels)

    self.in_channels = input_shape[-1]
      
    initializer = tf.initializers.RandomNormal( mean=0.0, stddev=1.0/self.lr_multiplier )
    self.w = self.add_weight(shape=[self.in_channels, self.units], initializer=initializer, trainable=True, name="FCL_kernel") # [in_channels, out_units]
    self.b = self.add_weight(shape=(self.units,), initializer=self.bias_init, trainable=True, name="FCL_bias") # [out_units]

    self.weight_gain = (self.gain * self.lr_multiplier) / tf.sqrt( tf.cast(self.in_channels, tf.float32) )
    super().build(input_shape)

  def call(self, x):

    w = tf.cast(self.weight_gain, self.w.dtype) * self.w
    b = self.b

    output = tf.add(tf.matmul(x, w), b) # (batch_size, in_channels) * (in_channels, out_units) -> (batch_size, out_units)
    
    if self.activation:
      return tf.nn.leaky_relu(output, alpha=0.2, name='FCL_activation') * self.lr_multiplier # ADDED * self.lr_multiplier 3/20/22
    else:
      return output * self.lr_multiplier

class Conv2dLayer(tf.keras.layers.Layer):

  def __init__(self,
               out_channels, 
               kernel_size=3, 
               bias=True,
               gain=2.0,
               lr_multiplier=1.0, # New below
               up=False, 
               down=False,
               resample_kernel=None,
               impl='ref',
               gpu=True,
               **kwargs):
    
    super(Conv2dLayer, self).__init__(**kwargs)
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.bias = bias
    self.pad = kernel_size != 1
    self.gain = gain
    self.lr_multiplier = lr_multiplier

    self.up = up
    self.down = down
    self.resample_kernel = resample_kernel
    self.impl = impl
    self.gpu = gpu

  def build(self, input_shape): # BS, H, W, C

    self.in_channels = input_shape[-1] # input channels
    self.weight_scale = (self.gain * self.lr_multiplier) / tf.sqrt( tf.cast(self.in_channels * self.kernel_size ** 2, tf.float32) )
    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
    self.w = self.add_weight(shape=[self.kernel_size, self.kernel_size, self.in_channels, self.out_channels], initializer=initializer, trainable=True, name="Conv2d_kernel") # [Kernel Size, Kernel Size, IN, OUT]
    super().build(input_shape)

  def call(self, input):

    w = tf.cast(self.weight_scale, self.w.dtype) * self.w

    if self.up:
          x = upsample_conv_2d(input, tf.cast(w, input.dtype), data_format='NHWC', k=self.resample_kernel, impl=self.impl, gpu=self.gpu)
    elif self.down:
          x = conv_downsample_2d(input, tf.cast(w, input.dtype), data_format='NHWC', k=self.resample_kernel, impl=self.impl, gpu=self.gpu)
    else:
          x = tf.nn.conv2d(input, tf.cast(w, input.dtype), data_format='NHWC', strides=[1, 1, 1, 1], padding='SAME')
    return x

