import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from Generator import Generator
from Discriminator import Discriminator
from Mapping_Network import Mapping


def step(values): # "hard sigmoid", useful for binary accuracy calculation from logits. negative values -> 0.0, positive values -> 1.0
    return 0.5 * (1.0 + tf.sign(values))


class AdaptiveAugmenter(tf.keras.Model):

    def __init__(self,
                 log2_resolution,
                 **kwargs):

        super(AdaptiveAugmenter, self).__init__(**kwargs)

        # stores the current probability of an image being augmented
        self.probability = tf.Variable(0.0)
        self.max_translation = 0.25 # 0.125
        self.max_rotation = 0.25 # 0.125
        self.max_zoom = 0.25
        self.target_accuracy = 0.85 # 0.85, 0.95
        self.integration_steps = 1500 # 1000, 2000
        self.augmenter = keras.Sequential([layers.InputLayer(input_shape=(2**log2_resolution, 2**log2_resolution, 3)),
                                           layers.RandomFlip("horizontal"),
                                           layers.RandomTranslation(height_factor=self.max_translation,
                                                                    width_factor=self.max_translation,
                                                                    interpolation="nearest"),
                                           layers.RandomRotation(factor=self.max_rotation, interpolation='nearest'),
                                           layers.RandomZoom(height_factor=(-self.max_zoom, 0.0), width_factor=(-self.max_zoom, 0.0), interpolation='nearest')], name="ada")
        
    def call(self, images, training):

        if training:
          
          batch_size = tf.shape(images)[0]
          augmented_images = self.augmenter(images, training)
          augmentation_values = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0) # Generate random numbers
          augmentation_bools = tf.math.less(augmentation_values, self.probability) # Get booleans in the indices where we want augmented images or not
          images = tf.where(augmentation_bools, augmented_images, images)

        return images

    def update(self, real_logits):

        current_accuracy = tf.reduce_mean(step(real_logits))
        # the augmentation probability is updated based on the dicriminator's accuracy on real images
        accuracy_error = current_accuracy - self.target_accuracy
        self.probability.assign( tf.clip_by_value(self.probability + accuracy_error / self.integration_steps, 0.0, 1.0) )


class StyleGAN2(tf.keras.Model):

  def __init__(self, log2_resolution, 
               batch_size,
               latent_dim,
               min_num_features=32,
               max_num_features=512,
               beta = 0.99,
               transformation_dict={},
               **kwargs):

    super(StyleGAN2, self).__init__(**kwargs)
    
    # Model Parameters
    self.log2_end_res = log2_resolution
    self.num_blocks = log2_resolution-1 # Because we are forcing it to start at 4 (or 2 in log2)
    self.batch_size = batch_size
    self.latent_dim = latent_dim
    self.max_num_features = max_num_features
    
    # Set Up Models
    self.discriminator = Discriminator(max_log2_res=log2_resolution, batch_size=batch_size, max_num_features=max_num_features, min_num_features=int(min_num_features*2))
    self.generator = Generator(log2_resolution, latent_dim=self.latent_dim, max_num_features=max_num_features, min_num_features=min_num_features)
    self.mapping_network = Mapping(num_blocks=self.num_blocks, batch_size=batch_size, latent_dim=self.latent_dim)
    self.ada = AdaptiveAugmenter(log2_resolution=log2_resolution)
    
    # Training Parameters
    self.gradient_penalty_coefficient = 10
    self.lazy_gradient_penalty_interval = 4
    self.lazy_path_penalty_interval = 32
    self.pl_weight = 2.0
    self.exp_sum_a = tf.Variable(0, dtype=tf.float32, trainable=False)
    self.train_step_counter = tf.Variable(0, dtype=tf.int32, trainable=False)
    self.beta = beta
    self.pl_decay = 1-beta
    # self.layer_name = self.name
  
  def generate_noise(self, batch_size=None):
    
    if batch_size==None:
      noise = [tf.random.normal((2, self.batch_size, 2 ** res, 2 ** res, 1)) for res in range(2, self.log2_end_res + 1)]
    else:
      noise = [tf.random.normal((2, batch_size, 2 ** res, 2 ** res, 1)) for res in range(2, self.log2_end_res + 1)]
    return noise

  def get_w(self, batch_size=None):
    
    if batch_size==None:
      z = tf.random.normal((self.batch_size, self.latent_dim))
      w_mapping = self.mapping_network(z)
    else:
      z = tf.random.normal((batch_size, self.latent_dim))
      w_mapping = self.mapping_network(z)

    return w_mapping

  def compile(self, steps_per_epoch, optimizer={}, metrics={}, *args, **kwargs):

    self.train_step_counter.assign(0)
    self.steps_per_epoch = steps_per_epoch

    self.discriminator_optimizer = optimizer.get('d_opt', tf.keras.optimizers.Adam(**{"learning_rate": 1e-3, "beta_1": 0.5, "beta_2": 0.99, "epsilon": 1e-8})) # CHANGED beta_1 from 0.0 to 0.5 3/21/22
    self.generator_optimizer = optimizer.get('g_opt', tf.keras.optimizers.Adam(**{"learning_rate": 1e-3, "beta_1": 0.5, "beta_2": 0.99, "epsilon": 1e-8})) # CHANGED beta_1 from 0.0 to 0.5 3/21/22

    self.discriminator_loss_metric = tf.keras.metrics.Mean(name="d_loss")
    self.generator_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    self.real_accuracy = keras.metrics.BinaryAccuracy(name="real_acc")
    self.generated_accuracy = keras.metrics.BinaryAccuracy(name="gen_acc")
    self.augmentation_probability_tracker = keras.metrics.Mean(name="aug_p")

    super(StyleGAN2, self).compile(*args, **kwargs)

  @property
  def metrics(self):
      return [self.discriminator_loss_metric, self.generator_loss_metric, 
              self.real_accuracy, self.generated_accuracy, self.augmentation_probability_tracker] # ADDED 3/21/22

  def adversarial_loss(self, output_realImgs_disc, output_genImgs_disc):

      real_labels = tf.ones(shape=(self.batch_size, 1)) # Should be ok bc train step updates self.batch_size
      generated_labels = tf.zeros(shape=(self.batch_size, 1))

      # the generator tries to produce images that the discriminator considers as real
      generator_loss = keras.losses.binary_crossentropy(real_labels, output_genImgs_disc, from_logits=True)

      # the discriminator tries to determine if images are real or generated
      discriminator_loss = keras.losses.binary_crossentropy(tf.concat([real_labels, generated_labels], axis=0),
                                                            tf.concat([output_realImgs_disc, output_genImgs_disc], axis=0), from_logits=True)
      return K.mean(generator_loss), K.mean(discriminator_loss)
      # return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)

  def gradient_penalty(self, reals, prev_discriminator_loss):

      with tf.GradientTape(watch_accessed_variables=False) as tape:
          tape.watch(reals)
          real_scores_out = self.discriminator(reals)
          real_grads = tape.gradient(tf.reduce_sum(real_scores_out), reals)
      gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
      discriminator_loss = gradient_penalty * (self.gradient_penalty_coefficient * 0.5) * 4
      return prev_discriminator_loss + K.mean(discriminator_loss)
      # return tf.reshape(reg,[-1,1])

  def path_length_penalty(self, const_input, prev_generator_loss):

      zs = tf.random.normal([self.batch_size, self.latent_dim])
      w_mapping_new = self.mapping_network(zs)
      noise = self.generate_noise()

      with tf.GradientTape(watch_accessed_variables=False) as tape:

          tape.watch(w_mapping_new)  
          fake_images_out = self.generator(const_input, w_mapping_new, noise, {})
          # Compute |J*y|.
          pl_noise = tf.random.normal(tf.shape(fake_images_out)) / np.sqrt(np.prod(fake_images_out.shape[2:]))
          lss = tf.reduce_sum(fake_images_out * pl_noise)
      # Get gradient norm
      gradients = tape.gradient(lss, w_mapping_new)
      gradient_norm = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(gradients), axis=2), axis=1))
      # Track exponential moving average of |J*y|.
      a = self.exp_sum_a + 0.01 * (tf.reduce_mean(gradient_norm) - self.exp_sum_a)
      self.exp_sum_a.assign(a)
      # Calculate (|J*y|-a)^2.
      pl_penalty = tf.square(gradient_norm - a)
      generator_loss = pl_penalty * self.pl_weight
      return prev_generator_loss + K.mean(generator_loss)
      # return tf.reshape(reg,[-1,1])

  def train_step(self, images_ds):

    self.batch_size = images_ds.shape[0]
    const_input = tf.ones((self.batch_size,) + (4, 4, self.max_num_features))
    z = tf.random.normal((self.batch_size, self.latent_dim))
    noise = self.generate_noise()

    apply_gradient_penalty = tf.equal(( self.train_step_counter ) % self.lazy_gradient_penalty_interval, 0) # Apply gradient penalty to discriminator_loss
    path_length_penalty = tf.math.greater_equal(self.train_step_counter, 5000) & tf.equal(( self.train_step_counter ) % self.lazy_path_penalty_interval, 0) # Apply path length penalty to generator_loss
    self.train_step_counter.assign_add(1)

    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

      # Generator
      w_mapping = self.mapping_network(z)
      generated_images = self.generator(const_input, w_mapping, noise, {})
      generator_weights = self.mapping_network.trainable_weights + self.generator.trainable_weights # trainable_weights, trainable_variables
      generated_images = self.ada(generated_images, training=True) # ADDED 3/21/22

      # Discriminator
      real_output = self.discriminator(images_ds)
      fake_output = self.discriminator(generated_images)

      #generator_loss, discriminator_loss = self.adversarial_loss(real_output, fake_output) # ADDED 3/21/22

      # Generator Loss
      generator_loss = K.mean(tf.nn.softplus(-fake_output)) # tf.nn.softplus(-fake_output) # K.mean(fake_output), tf.nn.softplus(-fake_output)

      # Discriminator Loss
      real_loss = tf.nn.softplus(-real_output) # K.mean(K.relu(1 - real_output))
      fake_loss = tf.nn.softplus(fake_output) # K.mean(K.relu(1 + fake_output))
      discriminator_loss = K.mean(real_loss) + K.mean(fake_loss) #real_loss + fake_loss

      discriminator_loss = tf.cond(apply_gradient_penalty, lambda: self.gradient_penalty(images_ds, discriminator_loss), lambda: discriminator_loss)
      generator_loss = tf.cond(path_length_penalty, lambda: self.path_length_penalty(const_input, generator_loss), lambda: generator_loss)

    # Gradients and Optimizers
    generator_gradients = generator_tape.gradient(generator_loss, generator_weights)
    self.generator_optimizer.apply_gradients(zip(generator_gradients, generator_weights))

    discriminator_gradients = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_weights) # trainable_weights, trainable_variables
    self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_weights)) # trainable_weights, trainable_variables

    # Update metrics
    self.discriminator_loss_metric.update_state(discriminator_loss)
    self.generator_loss_metric.update_state(generator_loss)
    self.ada.update(real_output)
    self.real_accuracy.update_state(1.0, step(real_output))
    self.generated_accuracy.update_state(0.0, step(fake_output))
    self.augmentation_probability_tracker.update_state(self.ada.probability)
    return {m.name: m.result() for m in self.metrics}

  def generate_images(self, batch_size=None, w_in=None, noise_in=None, transformation_dict={}):

    if batch_size==None:
      w_mapping = self.get_w() if w_in == None else w_in
      noise = self.generate_noise() if noise_in == None else noise_in
      const_input = tf.ones((self.batch_size,) + (4, 4, self.max_num_features))
    else:
      w_mapping = self.get_w(batch_size) if w_in == None else w_in
      noise = self.generate_noise(batch_size) if noise_in == None else noise_in
      const_input = tf.ones((batch_size,) + (4, 4, self.max_num_features))
    images = self.generator(const_input, w_mapping, noise, transformation_dict)
    return images
  
  def call(self, input):

    self.batch_size = input.shape[0]
    w_mapping = self.get_w()
    noise = self.generate_noise()
    const_input = tf.ones((self.batch_size,) + (4, 4, self.max_num_features))
    # transformation_dict = {'5': [ Translation(), 0.99 ], '2': [ Zoom(), 1.0 ]}
    images = self.generator(const_input, w_mapping, noise, {})
    return images
