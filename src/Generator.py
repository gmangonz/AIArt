from tensorflow.keras import layers
from tensorflow import keras
from Layers import FullyConnectedLayer
from Layers import ModulatedConv
from Network_Bending import NetworkBending
from dnnlib.ops.upfirdn_2d import upsample_2d

class toRGB(layers.Layer):

  """
  w - mapping put through affine transformation (self.to_style)
  Covolve x with style vector and add bias
  Run result through activation function
  
  """    

  def __init__(self, 
               filter_num=3, 
               **kwargs):
    
    super(toRGB, self).__init__(**kwargs)
    self.activation = layers.LeakyReLU(0.2)

  def build(self, input_shapes):

    x_shape, w_shape = input_shapes
    
    self.to_style = FullyConnectedLayer(x_shape[-1], bias='ones', name='FCL_toRGB', activation = True, gain= 1.0)
    self.conv = ModulatedConv(filters = 3, kernel_size = 1, padding = 'same', demod = False, name='ModConv_toRGB')
    self.bias = self.add_weight(shape=(3), initializer="zeros", trainable=True, name="toRGB_bias")
    super().build(input_shapes)

  def call(self, inputs):

    x, w = inputs
    style = self.to_style(w) # Output shape will have the same number of features as x
    x = self.conv([x, style]) # Will obtain the in_channels shape for both and will check if x and w has same shape and will return x with 3 channels
    x = x + self.bias

    return self.activation(x)


class StyleBlock(layers.Layer):
    
  """
  w - mapping put through affine transformation (self.to_style).
  (Modulated) Convolve x with style vector.
  Noise put through affine transformation (self.scale_noise).
  Add convolved x and scaled noise.
  Add bias.
  Run result through activation function.
  
  """    

  def __init__(self,
               out_channels,
               up = False,
               **kwargs):

    super(StyleBlock, self).__init__(**kwargs)
    self.out_channels = out_channels
    self.resample_kernel = [1,3,3,1]
    self.up = up
    self.conv = ModulatedConv(self.out_channels, kernel_size=3, up=self.up, name=f'StyleBlock_ModConv_{up}') # Will check if x and w has same shape | resample_kernel=self.resample_kernel, up=self.up
    self.activation = layers.LeakyReLU(0.2)

  def build(self, input_shapes):

    x_shape, w_shape, noise_shape = input_shapes

    assert noise_shape[-1] == 1, 'Noise should be single channel'

    self.desired_num_channels = x_shape[-1]
    self.to_style = FullyConnectedLayer(self.desired_num_channels, bias='ones', activation=True, lr_multiplier=1.0, gain=1.0, name='StyleBlock_FCL') # Output shape will have the same number of features as x # UPDATED 3/20/22

    self.scale_noise = self.add_weight(shape=[1, 1, 1, self.out_channels], initializer='zeros', trainable=True, name='scale_noise') # B for noise, previously zeros next ones
    self.bias = self.add_weight(shape=(self.out_channels,), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0), trainable=True, name="StyleBlock_bias") # previously zeros
    super().build(input_shapes)

  def call(self, inputs):

    x, w, noise = inputs

    s = self.to_style(w) # Will multiply w: (batch_size, in_channels) * (in_channels, x_shape[-1]) -> (batch_size, x_shape[-1])
    x = self.conv([x, s]) # Will obtain the in_channels shape for both which should be equal as stated above and will return x with self.out_channels
    x = x + self.scale_noise * noise # Even though noise is single channels, by multiplying it with [1, 1, 1, self.out_channels], it should broadcast to x's shape
    x = x + self.bias
  
    return self.activation(x)


class Generator_Block(layers.Layer):
  
  """
  The individual block of the StyleGAN2 Generator that performs:
    Takes image and upsamples, 
    Does twice: Takes w vector, performs affine transformation (self.to_style()), performs modulated 3x3 convolution, adds scaled noise and bias.
    Takes output and converts it to a 3-channel RGB image.
  
  """
  
  
  def __init__(self, 
               out_channels, 
               transformation_dict={},
               **kwargs):

    super(Generator_Block, self).__init__(**kwargs)

    self.layer_name = self.name
    self.style_block1 = StyleBlock(out_channels, name='StyleBlock_1', up=True)
    self.style_block2 = StyleBlock(out_channels, name='StyleBlock_2')
    self.networkbending = NetworkBending(self.layer_name)
    self.to_rgb = toRGB(name='to_RGB')

  def call(self, x, w, noise=[None, None], transformation_dict={}):

    x = self.style_block1([x, w, noise[0]])
    x = self.style_block2([x, w, noise[1]])
    rgb_img = self.to_rgb([x, w])
    rgb_img = self.networkbending([rgb_img, transformation_dict]) # UPDATED 3/20/22

    return x, rgb_img


class Generator(layers.Layer):

  def __init__(self,
               max_log2_res, 
               latent_dim=512,
               min_num_features=32,
               max_num_features=512,
               **kwargs):
    
    super(Generator, self).__init__(**kwargs)
    
    self.resample_kernel = [1, 3, 3, 1]
    out_channels_list = [min(max_num_features, min_num_features * (2 ** i)) for i in range(max_log2_res - 2, -1, -1)] # [min(512, 32 * (2 ** i)) for i in range(10 - 2, -1, -1)] -> [512, 512, 512, 512, 512, 256, 128, 64, 32]
    
    # Generate First Block that takes 4x4 constant input and only has 1 StyleBlock rather than 2.
    self.generator_input = layers.Input(shape=(4, 4, out_channels_list[0]), name='StyleGAN2_Input') # (bs, 4, 4, 512)
    self.style_block = StyleBlock(out_channels_list[0], name='First_StyleBlock')
    self.to_rgb = toRGB(name='First_toRGB')
    
    # Generate the Remaining Generator Blocks
    self.number_of_blocks = len(out_channels_list) # Should be max_log2_res - 1
    self.generator_blocks = [Generator_Block(out_channels_list[i]) for i in range(1, self.number_of_blocks)]
    
  def call(self, x, w, noise, transformation_dict={}):
    
    """
    x: input_img
    w: style inputs: (bs, number_of_blocks, 512)
    noise: list of input noise with length self.number_of_blocks; [(noise1a, noise1b), (noise2a, noise2b), (noise3a, noise3b) ...]
    transformation_dict: {'layer name': [Transformation, Magnitude]} for Network Bending
    
    """
    x = self.style_block([x, w[:, 0, :], noise[0][0]])
    rgb = self.to_rgb([x, w[:, 0, :]])

    for i in range(1, self.number_of_blocks): # Start at 1 since noise[0] and w[:, 0, :] have already been used, but that means use i-1 as index for generator_blocks list

      x, newRGB = self.generator_blocks[i-1](x, w[:, i, :], noise[i], transformation_dict)
      rgb = upsample_2d(rgb, k=self.resample_kernel, data_format='NHWC', impl='ref', gpu=True) + newRGB

    return rgb