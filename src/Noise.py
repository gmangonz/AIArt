import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from Parameters import base_resolutions

def get_base_noise(resolution):

    """
    For each resolution in [4, 8, 16, 32, 64, 128, 256, 512] set the seed get standard normal noise.
    Outpus shape is (res, res, 2). The reason for 2 channel noise is so that I can essentially create a vector and rotate it to create faux 3D noise.

    """

    rng   = np.random.default_rng(seed=resolution)
    theta = rng.standard_normal((resolution, resolution)) # rng.normal(0, sigma, (resolution, resolution))
    phi   = rng.standard_normal((resolution, resolution)) # rng.normal(0, sigma, (resolution, resolution))
    return np.stack( [theta, phi], axis=-1) # (res, res, 2)


def rotate_noise(noise, frame_value, sigma=1):

    """ 
    Given the noise of shape (res, res, 2) rotate the noise with frame_value and scale with amplitude factor of sigma essentially changing standard normal stddev to sigma. 
    Lower sigma = smoother noise.
    Higher sigma = sharper noise.

    Output shape: (res, res, 1)
    """

    frame = 2 * np.pi * frame_value # Between 0-360
    x_noise = noise[..., 0]
    y_noise = noise[..., 1]

    # Rotate x and y based on frame value to create faux 3D noise that I can control
    x = x_noise * np.cos(frame) - y_noise * np.sin(frame)
    y = x_noise * np.sin(frame) + y_noise * np.cos(frame)

    noise_image = x + y
    noise_image = noise_image[..., None]

    return noise_image * sigma


def get_xy_gradients(sigma, resolution=32):

    """ Will create random x and y values (between -1 & 1) for gradient vectors to create displacement map """

    # If resolution changes, then have a new seed
    rng   = np.random.default_rng(seed=resolution)
    theta = rng.normal(0, sigma, (resolution, resolution)) # rng.standard_normal((resolution, resolution))
    phi   = rng.normal(0, sigma, (resolution, resolution)) # rng.standard_normal((resolution, resolution))

    x_gradients = np.cos(np.pi * theta)           # 1 to -1
    y_gradients = np.sin(np.pi * phi + np.pi / 2) # 1 to -1

    return x_gradients, y_gradients
    return theta, phi


def make_displacement_map(frame, x_gradients, y_gradients, amplitude, shape=(512, 512)):

    """
    Make a displacement map for tf.image.dense_image_warp

    frame: comes from an audio signal that has values between 0 and 1. Helps smooth the transitions (assuming the signal is somewhat smooth)

    """

    frame = 2 * np.pi * frame # Between 0-360

    # Rotate gradients based on frame value
    x = x_gradients * np.cos(frame) - y_gradients * np.sin(frame)
    y = x_gradients * np.sin(frame) + y_gradients * np.cos(frame)

    # Cheap cloud noise generator
    gradients = np.stack( [x, y], axis=-1) # shape: (resolution, resolution, 2) of values between -1 and 1, i.e the gradient vectors
    gradients = tf.image.resize(gradients, size=shape, method='lanczos5') # shape: (shape[0], shape[1], 2)
    return gradients * amplitude


def get_displacement_map(frame_value, amplitude_factor, resolution_factor, shape=(512, 512)):

    """
    Inputs:
      frame_value:       between 0 and 1, will be mapped to 0-360 = will rotate the gradient vectors

      amplitude_factor:  between 0 and 1, will be mapped to 0-max_amplitude = Smaller - no displacement, larger - more displacement

      resolution_factor: between 0 and 1 = Lower - large clouds, Higher - small clouds

      shape: shape of noise used as inputs to StyleGAN2, one of (4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)

    Returns:

      displacement_map: displacement map for tfa.image.dense_image_warp, (shape[0], shape[1], 2)
    """
    resolution_factor_bins  = min(0.2 * (int(resolution_factor / 0.2) + 1), 1.0) # Bucketize to 5 levels: 0-0.19 -> 0.2, 0.2-0.39 -> 0.4, 0.4-0.59 -> 0.6, 0.6-0.79 -> 0.8, 0.8-1.0 -> 1.0
    base_resolution_min_max = base_resolutions.get(shape) # Get base resolution, i.e the max resolution of gradient's image size

    min_resolution = base_resolution_min_max[0]
    max_resolution = base_resolution_min_max[1]
    max_amplitude  = base_resolution_min_max[2]
    amplitude      = max(1, amplitude_factor*max_amplitude) # Amplitude should be at least 1
    resolution     = max(min_resolution, int(np.ceil(max_resolution*resolution_factor_bins)))

    x_gradients, y_gradients = get_xy_gradients(resolution=resolution, sigma=frame_value) # resolution
    displacement_map         = make_displacement_map(frame=frame_value, x_gradients=x_gradients, y_gradients=y_gradients, amplitude=amplitude, shape=shape)
    return x_gradients, y_gradients, displacement_map


@tf.function(input_signature=(tf.TensorSpec(shape=(1, None, None, 1), dtype=tf.float32), tf.TensorSpec(shape=(1, None, None, 2), dtype=tf.float32)))
def image_warp(image, flow):

    """ Run tfa.image.dense_image_warp """

    return tfa.image.dense_image_warp(image, flow)