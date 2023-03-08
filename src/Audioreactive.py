import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tqdm.autonotebook import tqdm
import subprocess
import scipy
from Network_Bending import Zoom, Translation, Rotation
from StyleGAN2 import StyleGAN2
import glob

class AudioReactive():

  def __init__(self, 
               max_features,
               noise,
               audio, 
               audio_file,
               train_ckpt_base_dir,
               ckpt_folder_name,
               ckpt_name,
               log2_resolution=9,
               melody=None, 
               drums=None, 
               bass=None, 
               sr=None,
               num_frames=None,
               setup=None,
               ckpt_weights=None,
               **kwargs):

    tf.keras.backend.clear_session()
    
    # Setup StyleGAN2
    self.max_features = max_features
    self.noise = noise
    self.stylegan2 = StyleGAN2(log2_resolution=log2_resolution, batch_size=None, latent_dim=512, min_num_features=16, max_num_features=max_features, name='StyleGAN2')
    self.stylegan2.compile(steps_per_epoch=1000, run_eagerly=False)
    self.stylegan2.build((2, 4, 4, max_features))
    self.stylegan2.load_weights(os.path.join(train_ckpt_base_dir, ckpt_folder_name, ckpt_name))

    # Get audio files and np.arrays
    self.audio_file = audio_file
    self.audio = audio
    self.melody = melody
    self.drums = drums
    self.bass = bass
    self.num_frames = num_frames
    self.sr = sr
    self.setup = setup
    
    # Set seeds
    self.melody_latents_seeds = [388, 51, 97, 251, 389, 447, 185, 35, 166, 8, 21, 39]
    self.drum_latents_seeds = [1, 39, 51, 97, 125, 166, 232, 251, 196, 320, 389]
    self.tempo_latents_seeds = [51, 139, 491, 93, 39, 186, 76] 
    
    # Get weights
    generator_weights_name = list(map(lambda x: x.name, self.stylegan2.generator.weights))
    generator_weights_value = list(map(lambda x: x.numpy(), self.stylegan2.generator.weights))
    mapping_weights_name = list(map(lambda x: x.name, self.stylegan2.mapping_network.weights))
    mapping_weights_value = list(map(lambda x: x.numpy(), self.stylegan2.mapping_network.weights))
    self.generator_weights = dict(zip(generator_weights_name, generator_weights_value))
    self.mapping_weights = dict(zip(mapping_weights_name, mapping_weights_value))
    
    # Load tensorflow model for super resolution
    SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
    self.upsample_model = hub.load(SAVED_MODEL_PATH)

  def generate_z(self, seeds=None):

    """
    Generate the 12 images that the music will be transitioning between.
    
    Output: (len(seeds), 512) where len(seeds) == 12.
    """

    if seeds == None:
      seeds = [176, 30, 35, 37, 0, 103, 60, 63, 92, 21, 104, 105] # [176, 60, 63, 92, 21, 30, 35, 37, 0, 103, 104, 105]
    zs = []
    for i in seeds:
        tf.random.set_seed(i)
        z_i = tf.random.normal((1, 512))
        zs.append(z_i)
    zs = tf.concat(zs, axis=0)
    return zs

  def interpolate_latents(self, latent_selection, n_frames, n_loops, loop=True):

    """
    Interpolate between the latents given.
    Args:
      latent_selection: (num, 1, 512)
      
    Output: (len(x), 1, 512) where len(x) == int(n_frames // max(1, n_loops))
    """

    if loop:
        latent_selection = np.concatenate( [ latent_selection, latent_selection[[0]] ] )

    x = np.linspace(0, 1, int(n_frames // max(1, n_loops)))
    base_latents = np.zeros((len(x), 1, 512))
    for lat in range(512):
        try:
            tck = scipy.interpolate.splrep(np.linspace(0, 1, latent_selection.shape[0]), latent_selection[:, 0, lat]) # x: (0, 0.5, 1) y: (a, b, c) that are across the first dimension
            base_latents[:, 0, lat] = scipy.interpolate.splev(x, tck)
        except:
            base_latents[:, 0, lat] = np.interp(np.linspace(0, 1, int(n_frames // max(1, n_loops))), [0, 1], latent_selection[:, 0, lat])

    return base_latents

  def get_latents_from_impulse(self, impulse_idx, transition_length=24, seeds=None):
    
    """
    Using a list of timestamps, I use these timestamps to indicate when to change latents.
    Using a transition_length, this will allow for a smoother interpolated transition between latents.
    
    Output: (num_frames, 1, 512)
    """

    if impulse_idx[0] != 0:
        impulse_idx = np.insert(impulse_idx, 0, 0)
    if impulse_idx[-1] >= self.num_frames:
        impulse_idx[-1] = self.num_frames - 1
    if impulse_idx[-1] != self.num_frames-1:
        impulse_idx = np.append(impulse_idx, self.num_frames-1)

    # Set seeds
    num_impulses = len(impulse_idx) - 1
    seeds = [1, 8, 21, 35, 39, 51, 97, 125, 166, 232, 251, 364, 323, 388, 389] if seeds == None else seeds
    
    # In the case there are more impulses than number of seeds to use, repeat the seeds accordingly
    if num_impulses > len(seeds):
        seeds = seeds * int(np.ceil(num_impulses/len(seeds)))

    # Get zs and w_custom_single
    zs = self.generate_z(seeds) # (len(seeds), 512)
    w_custom = self.stylegan2.mapping_network(zs) # (len(seeds), 8, 512)
    w_custom_single = w_custom[:, 0, :][:, None, :] # (len(seeds), 1, 512)

    # Set up latents
    latents = np.zeros(shape=(self.num_frames, 1, 512))

    # Set new latents after an impulse
    length = len(impulse_idx[:-1])
    for i in range(length):
        start = int(impulse_idx[i])
        end = int(impulse_idx[i+1])
        latents[start: end, :, :] = w_custom_single[i]

    # Make transition between latents
    for i in impulse_idx[1:-1]: # Do not include 0 and last timestamp as there cannot be a transition before or afterwards as they are the start and end.
        latent_0 = latents[[i-1]] # (1, 1, 512)
        latent_1 = latents[[i+1]] # (1, 1, 512)
        transition = self.interpolate_latents(tf.concat([latent_0, latent_1], axis=0), n_frames=transition_length, n_loops=1, loop=False)
        latents[i - min(5, i): i + transition_length-min(5, i)] = transition

    return latents

  def get_latents(self):
    
    """
    Using tempo and drums, make latent sequence using their impulses.
    Make chroma weighted latents.
    
    Output: (num_frames, 8, 512)
    """

    # Tempo Latents from its impulse signal
    tempo_latents = self.get_latents_from_impulse(self.setup.tempo_segments, seeds=self.tempo_latents_seeds) # (num_frames, 1, 512)
    tempo_latents = tf.cast(tf.tile(tempo_latents, [1, 5, 1]), tf.float32) 

    # Drum Latents from its impulse signal
    impulse = self.setup.drums_onsetsHigh_peaks + self.setup.drums_onsetsLow_peaks + self.setup.drums_onsetsMid_peaks
    drum_latents = self.get_latents_from_impulse(np.nonzero(impulse)[0], transition_length=10, seeds=self.drum_latents_seeds) # (num_frames, 1, 512)
    drum_latents = tf.cast(tf.tile(drum_latents, [1, 2, 1]), tf.float32) 

    # Get mapping from seeds
    zs = self.generate_z(self.melody_latents_seeds) # (len(seeds), 512)
    w_custom = self.stylegan2.mapping_network(zs) # (len(seeds), 8, 512)
    w_custom_single = w_custom[:, 0, :] # (len(seeds), 512)

    # Chroma - Melody Latents
    chroma_latents_melody = self.setup.melody_chroma @ w_custom_single # (n, 12) @ (12, 512) -> (n, 512) of weighted latents
    chroma_latents_melody = tf.tile(chroma_latents_melody[:, None, :], [1, 1, 1]) # (n, 2, 512)

    return tf.concat([chroma_latents_melody, tempo_latents, drum_latents], axis=1) # <- fav so far 

  def get_noise(self):
    
    """
    Using drums, get a weighted sum of the noise and gaussian noise. Because 1 (not really tho) indicates a drum hit and 0 no drum hit, 
    when a drum hits, noise transitions from smooth gaussian noise to more "noisy" noise and back to gaussian noise as the drum hit signal dies down to 0.
    Recall self.setup.drums_onsetsMid_2 looks like half right-side gaussians of length radius_1 in Setup.setup
    
    Append and tile the weighted noise to match the required input for noise of StyleGAN2.
    
    I use i+2 to match the log2 resolution steps from 2 (4 resolution) to 9 (512 resolution).
    
    Output: 7 * (2, num_frames, res, res, 1)
    """

    noise_list = []

    for i, (noise, gauss_noise) in enumerate(self.noise): # 7 * ((n, res, res, 1), (n, res, res, 1))
      
      if i+2 <= 5:
        weighted_noise = self.setup.drums_onsetsMid_2[..., None, None, None] * noise + (1-self.setup.drums_onsetsMid_2[..., None, None, None]) * gauss_noise  
      if i+2 > 5:
        weighted_noise = self.setup.drums_onsetsHigh_2[..., None, None, None] * noise + (1-self.setup.drums_onsetsHigh_2[..., None, None, None]) * gauss_noise
      
      weighted_noise /= weighted_noise.std() * 2
      noise_list.append(np.tile(weighted_noise[None, ...], (2, 1, 1, 1, 1)))

    return noise_list

  def noise_transforms(self, noise, num):
    
    """
    Transform each noise input and concat.
    
    """
    
    noise_a = noise[0] # (1, res, res, channel)
    noise_b = noise[1] # (1, res, res, channel)
    noise_a = tf.keras.layers.RandomTranslation(height_factor=(0, 0), width_factor=(num, num), fill_mode="constant", fill_value=1)(noise_a)
    noise_b = tf.keras.layers.RandomTranslation(height_factor=(0, 0), width_factor=(num, num), fill_mode="constant", fill_value=1)(noise_b)
    return tf.concat([noise_a[None, ...], noise_b[None, ...]], 0) # (2, 1, res, res, channel)

  def generate_video(self, img_loc, out_loc, super_res=False):
    
    """
    Make video with the various inputs setup.
    
    """

    assert os.path.isdir(img_loc) == True, f'{img_loc} does not exist'
    assert os.path.isdir(out_loc) == True, f'{out_loc} does not exist'

    print('Getting Preperations...')
    latents = self.get_latents() # (n, 8, 512)
    noises = self.get_noise() # 7 * (2, n, res, res, 1) | # noise_in = [tf.random.normal((2, 1, 2 ** res, 2 ** res, 1)) for res in range(2, self.log2_resolution + 1)]

    print('Generating Images...')
    for frame in tqdm(range(self.num_frames)):

      # Get input for frames
      w_in = latents[frame][None, ...] # (1, 8, 512)
      noise_in = [self.noise_transforms( noise_block[:, frame, :, :, :][:, None, :, :, :], self.setup.drums_onsetsHigh_2[frame] ) for noise_block in noises]
      noise_in = noise_in + [ tf.tile(scipy.ndimage.gaussian_filter( tf.random.normal((1, 512, 512, 1), stddev=1), 5 )[None,...], [2, 1, 1, 1, 1]) ] # 8 * (2, n, res, res, 1): Add final gaussian filtered 512-resolution noise

      # Network Bending: Zoom, Translation, Rotation, Contrast
      transformation_dict = {'generator__block_6': [ Zoom(), self.setup.drums_onsetsMid_2_r_p[frame] ], 'generator__block_3': [ Zoom(), self.setup.bass_onsetsHigh_2_r_p[frame] ]}

      # Feed model and save result
      custom_img = self.stylegan2.generate_images(batch_size=1, w_in=w_in, noise_in=noise_in, transformation_dict=transformation_dict)
      plt.imsave(os.path.join(img_loc, 'img_{num:0{width}}.jpg'.format(num=frame, width=4)), tf.clip_by_value(custom_img[0], 0, 1).numpy())
    
    print(f'Collecting Images with Upscaling Set to {super_res} and Writing Video...')
    files = glob.glob(os.path.join(img_loc, '*.jpg'))
    fps = 24
    out = None

    for img_file in tqdm(files):

        img = cv2.imread(img_file)
        if super_res:
            scaled_img = ((img - img.mean())/(img.std() + 1e-9)).astype(np.float32) # Zero mean and set std to 1.
            super_res_img = self.upsample_model(scaled_img[None, ...]) # Get super resolution image.
            fixed_img = (super_res_img - K.mean(super_res_img))/(K.std(super_res_img) + 1e-9) # Zero mean output image and set output image's std to 1.
            img = (((fixed_img*img.std()) + img.mean())[0]).numpy() # Add original mean to output image and set output image's std to original std. (This makes it so color grading is easier, i think idk tho)
            img = np.clip(img/255, 0, 1)

        height, width, layers = img.shape
        size = (width, height)
        if out == None:
            out = cv2.VideoWriter(os.path.join(out_loc, 'stylegan2.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        if super_res:
            out.write((img*255).astype(np.uint8))
        else:
            out.write(img)
    out.release()

    print('Adding Audio to Video...')
    subprocess.call(['ffmpeg', '-i', os.path.join(out_loc, 'stylegan2.mp4'), '-i', self.audio_file, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', os.path.join(out_loc, 'stylegan2_audio_reactive.mp4')])

# Things removed

# In def __init___()
    # self.ckpt_weights = ckpt_weights
    # self.interpolate_weights()

# In def get_latents()
    # return tf.concat([tempo_latents, drum_latents, chroma_latents_melody], axis=1)
    # return tf.concat([tempo_latents, drum_latents], axis=1)
    # return tf.concat([drum_latents, chroma_latents_melody, tempo_latents], axis=1)
    # return tf.concat([tempo_latents, chroma_latents_melody, drum_latents], axis=1)

  # def interpolate_weights(self):

  #   interpolates = {}
  #   for name, weight in self.generator_weights.items():
  #     if 'StyleBlock_FCL/FCL_kernel' in name:
  #       weights_a = self.ckpt_weights['4_17_22'][name]
  #       weigths_b = self.ckpt_weights['4_15_22'][name] # 3_29_22, 4_1_22, 4_15_22
  #       interpolate_btwn_weights = np.linspace(weights_a, weigths_b, 30)
  #       interpolates[name] = interpolate_btwn_weights
  #   self.interpolates = interpolates

  # def update_weights(self, num):

  #   # FCL_bias: (512,), FCL_kernel: (512, 512)
  #   new_weights = []
  #   for name, weight in self.generator_weights.items():
  #     if 'StyleBlock_FCL/FCL_bias' in name:
  #       new_weights.append(weight)
  #     elif 'StyleBlock_FCL/FCL_kernel' in name:
  #       # corresponding_interpolated_weights = self.interpolates[name]
  #       # if num > 0.01:
  #       #   idx = int((1 - num) * 30)
  #       #   new_weights.append(corresponding_interpolated_weights[idx])
  #       # else:
  #         new_weights.append(weight)
  #     else:
  #       new_weights.append(weight)
  #   return new_weights
  
      # Update Weights
      # new_weights = self.update_weights(self.setup.bass_onsetsHigh_2[frame])
      # self.stylegan2.generator.set_weights(new_weights)