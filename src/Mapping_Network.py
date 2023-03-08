import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from Layers import FullyConnectedLayer

class Mapping(layers.Layer):

  def __init__(self,
               num_blocks,
               batch_size=None,
               latent_dim=512,
               num_layers=8,
               **kwargs):
    
    super(Mapping, self).__init__(**kwargs)
    z = layers.Input(shape=(latent_dim), batch_size=batch_size)
    w = layers.Lambda(lambda x: x / tf.math.sqrt(tf.reduce_mean(x ** 2, axis=1, keepdims=True) + 1e-8))(z) # Pixel Norm # UPDATED FROM -1 to 1
    
    for i in range(num_layers):
        w = FullyConnectedLayer(latent_dim, bias='ones', activation=False, lr_multiplier=0.5, name=f'Mapping_FCL_{i}')(w)
        w = layers.LeakyReLU(0.2)(w)
    w = tf.tile(tf.expand_dims(w, 1), (1, num_blocks, 1)) # (None, 512) -> (None, 1, 512) -> (None, num_blocks, 512)

    self.mapping_net = keras.Model(z, w, name="Mapping_Network")

  def call(self, z):

    return self.mapping_net(z)