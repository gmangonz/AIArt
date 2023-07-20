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
import glob
import tensorflow_addons as tfa
import librosa
from functools import reduce
from functools import partial

from Network_Bending import DisplacementMap, Zoom
from StyleGAN2 import StyleGAN2
from Parameters import params, num_frames, zoom_params
from Setup import MusicProcessing
from Noise import rotate_noise, image_warp, get_base_noise, get_displacement_map
from Music_Processing import normalize, make_plateu

def get_seeds(name, input_list):

    """
    Will generate a list of seeds that match the length of input list

    """
    rng   = np.random.default_rng(seed=params.get(name))
    seeds = rng.integers(low=0, high=500, size=(len(input_list)))
    return list(seeds)


def generate_z(seeds=None):

    """
    Generate the 12 images that the music will be transitioning between.

    Output: (len(seeds), 512) where len(seeds) == 12.
    """

    if seeds == None:
      seeds = [176, 30, 35, 37, 0, 103, 60, 63, 92, 21, 104, 105]
    zs = []
    for i in seeds:
        tf.random.set_seed(i)
        z_i = tf.random.normal((1, 512))
        zs.append(z_i)
    zs = tf.concat(zs, axis=0)
    return zs


def generate_sawtooth(impulse_idxs, num_frames, rms_signal=None):
    
    """
    Given the indicies indicating tempo changes, iterate through start and end pairs to generate a signal that goes from 0 to 1.
    This signal can be scaled depending on the RMS seen between the start and end points. 

    This signal will be used in Noise.rotate_noise and will be mapped from 0-1 to 0-360 degrees. Essentially creating faux 3D noise.
    

    Output:
      out: (num_frames,)
    """

    scale = 1
    out = np.zeros(num_frames)
    for i in range(1, len(impulse_idxs)):
      start = impulse_idxs[i-1]
      end = impulse_idxs[i]
      if isinstance(rms_signal, np.ndarray):
          scale = rms_signal[start: end+1].mean()
      out[start: end+1] = np.linspace(0, 1, end+1-start) * scale
    return out


@tf.function(input_signature=(tf.TensorSpec(shape=(None, 1, 512), dtype=tf.float32), ))
def safe_normalize(v):
    
    """ Normalizes the input tensor. Safegaurds against dividing by zero. Returns tensor of same shape. """

    norm      = tf.clip_by_value(tf.norm(v, axis=-1, keepdims = True), 1e-36, 1e36)
    norm_safe = tf.logical_and(tf.not_equal(norm, 0.), tf.logical_not(tf.math.is_nan(norm)))
    norm      = tf.where(norm_safe, norm, tf.ones_like(norm))
    return v * tf.where(norm_safe, 1.0/norm, tf.zeros_like(norm))


@tf.function(input_signature=(tf.TensorSpec(shape=(1, 1, 512), dtype=tf.float32),
                              tf.TensorSpec(shape=(1, 1, 512), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, ), dtype=tf.float32)))
def slerp(v0, v1, t):
    """
    Interpolate between v0 and v1.
    Inputs:
      v0: (1, 1, 512)
      v1: (1, 1, 512)
      t:  (num_frames, )

    Outputs:
    out: (len(t), 1, 1) (len(x), 1, 512) where len(t) == int(num_frames // max(1, n_loops))
    """
    t   = t[..., None, None] # (num_frames, 1, 1)
    dot = tf.reduce_sum(safe_normalize(v0)*safe_normalize(v1), axis=-1, keepdims=True) # (1, 1, 1)

    signflip  = tf.where(tf.less_equal(dot, 0.), -1.*tf.ones_like(dot), tf.ones_like(dot))
    v1       *= signflip
    dot      *= signflip
    sdot      = tf.clip_by_value(dot, -1.0, 1.0)
    omega     = tf.acos(sdot) # (1, 1, 1)
    sin_omega = tf.sin(omega) # (1, 1, 1)

    s0 = tf.sin((1 - t) * omega) / (sin_omega+1e-19)
    s1 = tf.sin(omega * t)       / (sin_omega+1e-19)
    sq = (s0 * v0) + (s1 * v1)
    return sq


@tf.function(input_signature=(tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32), ))
def gauss_blur(img):

    """ Guassian blur 1-channel image """

    size  = 2
    sigma = 1
    x_range = tf.range(-(size-1)//2, (size-1)//2 + 1, 1)
    y_range = tf.range((size-1)//2, -(size-1)//2 - 1, -1)

    xs, ys = tf.meshgrid(x_range, y_range)
    r      = tf.cast(xs**2 + ys**2, tf.float32)
    exp    = tf.exp(-(r)/(2*(sigma**2)))
    kernel = exp / (2*np.pi*(sigma**2))
    kernel = tf.cast( kernel / tf.reduce_sum(kernel), tf.float32)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1) # filter/kernel should be a tensor of shape [filter_height, filter_width, in_channels, out_channels]

    blur_image = tf.nn.conv2d(img[None, ...], filters=kernel, strides=[1,1,1,1], padding='SAME', name='r_blur')
    blur_image = tf.cast(blur_image, img.dtype)
    return blur_image[0]


@tf.function(input_signature=(tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)))
def get_comb_noise(cloud_noise_image, noise_image):

    # Get new noise and gauss filter
    combined_noise = cloud_noise_image + noise_image
    combined_noise = gauss_blur(combined_noise) # tfa.image.gaussian_filter2d(combined_noise, filter_shape=5, sigma=3)

    # Normalize new noise between -1 and 1
    a = -1
    b =  1
    A = tf.reduce_min(combined_noise)
    B = tf.reduce_max(combined_noise)

    C = (a-b + 1e-6)/(A-B + 1e-6)
    k = (C*A - a)/(C)

    return (combined_noise-k)*C


class AudioReactive(MusicProcessing):

  def __init__(self, max_features, log2_resolution, train_ckpt_base_dir, ckpt_folder_name, ckpt_name, params,
               dict_config,
               **kwargs):

    tf.keras.backend.clear_session()
    if isinstance(dict_config, list):
      dict_config = dict_config[0]
    super(AudioReactive, self).__init__(dict_config)
    super(AudioReactive, self).setup(params)

    # Setup StyleGAN2
    self.max_features = max_features
    self.stylegan2 = StyleGAN2(log2_resolution=log2_resolution, batch_size=None, latent_dim=512, min_num_features=16, max_num_features=max_features, name='StyleGAN2')
    self.stylegan2.compile(steps_per_epoch=1000, run_eagerly=False)
    self.stylegan2.build((2, 4, 4, max_features))
    checkpoint = os.path.join(train_ckpt_base_dir, ckpt_folder_name, ckpt_name)
    print(f'loading weights from {checkpoint}')
    self.stylegan2.load_weights(checkpoint)

    # Load tensorflow model for super resolution
    SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
    self.upsample_model = hub.load(SAVED_MODEL_PATH)

  def get_latents_from_impulse(self, impulse_idx, transition_length=24, seeds=None):

    """
    Using a list of timestamps, I use these timestamps to indicate when to change latents.
    Using a transition_length, this will allow for a smoother interpolated transition between latents.

    Output: (num_frames, 1, 512)
    """

    # Set seeds
    num_impulses = len(impulse_idx) - 1
    seeds = [1, 8, 21, 35, 39, 51, 97, 125, 166, 232, 251, 364, 323, 388, 389] if seeds == None else seeds

    # In the case there are more impulses than number of seeds to use, repeat the seeds accordingly
    if len(seeds) < num_impulses:
        seeds = seeds * int(np.ceil(num_impulses/len(seeds)))

    # Get zs and w_custom_single
    zs = generate_z(seeds) # (len(seeds), 512)
    w_custom_single = self.stylegan2.mapping_network(zs)[:, 0, :][:, None, :] # (len(seeds), 1, 512), remember they are tiled, so we only need 1

    # Set up latents
    latents = np.zeros(shape=(self.num_frames, 1, 512))

    # Make transition between latents
    for i in range(0, num_impulses):
      start = int(impulse_idx[i])
      end   = int(impulse_idx[i+1])
      latents[start: end, :, :] = w_custom_single[i]
      if i > 0:
        latent_from = w_custom_single[i-1][None, ...]
        latent_to   = w_custom_single[i][None, ...]
        transition  = slerp(latent_from, latent_to, tf.linspace(0.0, 1.0, transition_length)) # interpolate_latents(tf.concat([latent_from, latent_to], axis=0), n_frames=transition_length, n_loops=1, loop=False)
        start_idx   = start - min(5, start) # Safegaurds going below 0
        end_idx     = min(start + transition_length-min(5, start), end) # Safegaurds going above max
        latents[start_idx: end_idx, :, :] = transition.numpy()[: end_idx-start_idx ] # Allows for correct broadcasting

    return latents

  def get_chroma_weighted_latents(self, seeds, chroma):

    zs = generate_z(seeds)                        # (len(seeds), 512)
    w_custom = self.stylegan2.mapping_network(zs) # (len(seeds), 8, 512)\\images\\logo-vertical.svg
    w_custom_single = w_custom[:, 0, :]           # (len(seeds), 512)
    chroma_latents  = chroma @ w_custom_single    # (n, 12) @ (12, 512) -> (n, 512) of weighted latents
    chroma_latents  = chroma_latents[:, None, :]  # (n, 1, 512)

    return chroma_latents

  def get_latents(self):

    """
    Using tempo and drums, make latent sequence using their impulses.
    Make chroma weighted latents.

    Output: (num_frames, 8, 512)
    """

    # Tempo Latents from its impulse signal
    tempo_latents = self.get_latents_from_impulse( self.segmentation_idxs, transition_length=24, seeds=get_seeds('tempo_latents_seed', self.segmentation_idxs) ) # (num_frames, 1, 512)
    tempo_latents_B = self.get_latents_from_impulse( self.segmentation_idxs, transition_length=10, seeds=get_seeds('drum_latents_seed', self.segmentation_idxs) ) # (num_frames, 1, 512)

    # Get melody chroma weighted latents
    m = 2
    chroma_latents_melody   = self.get_chroma_weighted_latents(self.melody_latents_seeds, self.audio_chroma) # (n, 1, 512) audio_chroma, melody_chroma
    tempo_latents           = tf.cast(tf.tile(tempo_latents          , [1, 6, 1]), tf.float32)
    tempo_latents_B         = tf.cast(tf.tile(tempo_latents_B        , [1, m, 1]), tf.float32)
    chroma_latents_melody   = tf.cast(tf.tile(chroma_latents_melody  , [1, 2, 1]), tf.float32)

    return tf.concat([chroma_latents_melody, tempo_latents], axis=1), tf.concat([chroma_latents_melody, tempo_latents_B, tempo_latents[:, 0:-m,:]], axis=1)

  def get_noise(self, noise, disp_maps, frame_value, sigma=1):

    """
    Given noise list, I create faux noise with frame_value that rotates the noise vectors. Sigma controls the amplitude of the noise (stddev) which smoothens the noise.


    Append and tile the weighted noise to match the required input for noise of StyleGAN2.
    I use i+2 to match the log2 resolution which steps from 2 (4 resolution) to 9 (512 resolution).

    """
    noise_list = []
    for i, (noise_res, disp_map) in enumerate(zip(noise, disp_maps)):
      if i+2 < 6:
        noise_res = rotate_noise(noise_res, frame_value, sigma=1)[None, ...] # (1, 2**res, 2**res, 1)
        noise_list.append( tf.tile(noise_res[None, ...], [2, 1, 1, 1, 1]) )
      else: # 6, 7, 8, 9
        noise_res = rotate_noise(noise_res, frame_value, sigma=sigma)[None, ...] # (1, 2**res, 2**res, 1)
        noise_res = image_warp(noise_res, disp_map[None, ...])
        noise_list.append( tf.tile(noise_res[None, ...], [2, 1, 1, 1, 1]) )
    return noise_list

  def generate_video(self, img_loc, out_loc, audio_file, super_res=False):

    """
    Make video with the various inputs.

    """

    assert os.path.isdir(img_loc) == True, f'{img_loc} does not exist'
    assert os.path.isdir(out_loc) == True, f'{out_loc} does not exist'

    def compose(f_in, g_in):
        return lambda x : g_in(f_in(x))

    print('Getting Preperations...')
    # Get latents and noise
    latents_A, latents_B = self.get_latents()                               # (n, 8, 512)
    noise_list           = [get_base_noise(2**res) for res in range(2, 10)] # 8 x (2**res, 2**res, 2)

    # Prepare Sawtooth signal and sigma 
    rms_signal           = scipy.signal.resample(scipy.ndimage.gaussian_filter1d(librosa.feature.rms(y=self.drums)[0], 64, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0), num_frames)
    rms_signal_for_noise = normalize(rms_signal, 0.4, 1.2) # Less RMS -> Smoother, More -> Sharper
    sawtooth_signal      = generate_sawtooth(self.segmentation_idxs, self.num_frames, normalize(rms_signal, 1, 3)) # sawtooth wave, in self.get_noise -> rotate_noise will be mapped to 0-360

    print('Generating Images...')
    for frame in tqdm(range(self.num_frames)):

      # Get input for frames
      w_in      = (1 - self.drums_rms[frame]) * latents_A[frame][None, ...] + (self.drums_rms[frame]) * latents_B[frame][None, ...] # (1, 8, 512) drums_rms, noise_0
      disp_maps = [get_displacement_map(frame_value=sawtooth_signal[frame], amplitude_factor=self.drums_rms[frame], resolution_factor=0.9, shape=(2**res, 2**res))[-1] for res in range(2, 10)] # drums_rms, noise_0
      noise_in  = self.get_noise(noise_list, disp_maps, frame_value=sawtooth_signal[frame], sigma=rms_signal_for_noise[frame]) # 1-self.drums_rms[frame], noise_0

      # Network Bending
      transformation_dict = {}
      for layer_name, (functions, frame_values) in self.transformation_dict.items():
        
        # Just in case; Zoom requires to go from 1 to 0 as 1 is no zoom
        frame_values_zoom = make_plateu(frame_values, zoom_params[layer_name]) 

        composed_funcs = [lambda x: x] # Have at least 1 function so functools.reduce works
        for func in list(functions):   # Will make non list to a list, else will keep same list

            if isinstance(func, DisplacementMap):
                gen_block_num = ''.join( filter(str.isdigit, layer_name) )
                func = partial( func, flow=disp_maps[ int(gen_block_num)+1 ] )
            if isinstance(func, Zoom):
                func = partial( func, frame_value=frame_values_zoom[frame] )
            else:
                func = partial( func, frame_value=frame_values[frame] )
          
            composed_funcs.append(func)
        transformation_dict[layer_name] = reduce(compose, composed_funcs) # Just need to take an image as input and this will cascade through the functions

      # Feed model
      custom_img = self.stylegan2.generate_images(batch_size=1, w_in=w_in, noise_in=noise_in, transformation_dict=transformation_dict)
      custom_img = normalize(custom_img, 0, 1)[0]

      # Save result
      plt.imsave(os.path.join(img_loc, 'img_{num:0{width}}.jpg'.format(num=frame, width=4)), np.clip(custom_img.numpy(), 0.0, 1.0))

    print(f'Collecting Images with Upscaling Set to {super_res} and Writing Video...')
    files = glob.glob(os.path.join(img_loc, '*.jpg'))
    fps = 24
    out = None

    for img_file in tqdm(files):

        img = cv2.imread(img_file)
        if super_res:
            scaled_img    = ((img - img.mean())/(img.std() + 1e-9)).astype(np.float32)            # Zero mean and set std to 1.
            super_res_img = self.upsample_model(scaled_img[None, ...])                            # Get super resolution image.
            fixed_img     = (super_res_img - K.mean(super_res_img))/(K.std(super_res_img) + 1e-9) # Zero mean output image and set output image's std to 1.
            img           = (((fixed_img*img.std()) + img.mean())[0]).numpy()                     # Add original mean to output image and set output image's std to original std. (This makes it so color grading is easier, i think idk tho)
            img           = np.clip(img/255, 0, 1)

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
    subprocess.call(['ffmpeg', '-i', os.path.join(out_loc, 'stylegan2.mp4'), '-i', audio_file, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', os.path.join(out_loc, 'stylegan2_audio_reactive.mp4')])