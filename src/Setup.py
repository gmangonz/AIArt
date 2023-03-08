import numpy as np
from scipy import signal
import librosa.display
import ruptures as rpt
import scipy
import tensorflow as tf
from tqdm.autonotebook import tqdm

class SetUp():

  def __init__(self,
               audio,
               duration,
               melody=None,
               drums=None,
               bass=None,
               sr=None,
               num_frames=None):

      self.audio = audio
      self.duration = duration
      self.melody = melody
      self.drums = drums
      self.bass = bass
      self.num_frames = num_frames
      self.sr = sr

  def get_onsets(self, audio, sr, n_frames, margin=3, power=1, fmin=20, fmax=8000, smooth=1):

      """
      Gets the onset of the percussive aspects of given audio; detects changes in audio.
      Resamples the signal to the desired frames where n_frames = duration*fps.
      Clips the resample to orignal min and max.
      Gaussian filter the signal to reduce noise.
      Normalize.
      
      """
  
      y_perc = librosa.effects.percussive(y=audio, margin=margin)
      onset = librosa.onset.onset_strength(y=y_perc, sr=sr, fmin=fmin, fmax=fmax, hop_length=128)
      onset = np.clip(signal.resample(onset, n_frames), onset.min(), onset.max())
      onset = scipy.ndimage.gaussian_filter1d(onset, smooth, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
      onset = (onset - np.min(onset))/np.ptp(onset)
      return onset ** power

  def get_chroma(self, audio, sr, n_frames, notes=12, margin=3, use_harm=True, smooth=1):

      """
      Gets the chromagram of the harmonic aspects of given audio; expresses the audio's pitch content over the twelve chroma bands.
      Gaussian filter the rows.
      Resample the image to the desired frames where n_frames = duration*fps.
      Normalize so that rows adds to 1. Why rows? Shape is # (num_frames, 12), when we grab a frame we get (12,) and we want this array to add to 1.
      Sort the rows based on their sum, largest on top.
      
      """
  
      y_harm = librosa.effects.harmonic(y=audio, margin=margin) if use_harm == True else audio
      chroma = librosa.feature.chroma_cens(y=y_harm, sr=sr, hop_length=128) # (12, 10336)
      chroma = scipy.ndimage.gaussian_filter1d(chroma.T, 30, axis=0, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0).T
      chroma = signal.resample(chroma.T, n_frames) # (num_frames, 12)
  
      chrom_class_norm = chroma/chroma.sum(axis = 1, keepdims = 1) # (num_frames, 12)
      chrom_class_sum = np.sum(chrom_class_norm, axis=0)
      chroma_sort = np.argsort(chrom_class_sum)
      chroma = chrom_class_norm[:, chroma_sort]
  
      return tf.cast(chroma, tf.float32)

  def tempo_segmentation(self):
      
      """
      Using the ruptures package, segment the audio with timestamps where there are tempo changes (predicted).
      Adjust so that the timestamps match the desired number of frames.
      Add 0 at start and make last timeframe equal num_frames-1.
      
      """
      # return np.array([0, 590, 719])
      # return np.array([0, 590, 912, 1320, 1632, 2159])
      return np.array([0, 592, 1188, 1776, 2368, 2952, 3552, 4142, 4183, 4238, 4232, 4335, 4377, 4432, 4473, 4526, 4928, 5289, 6152, 6201, 6253, 6297, 6350, 6393, 6446, 7272, 7600, 7840, 7904, 8496, 8581, 9202, 9800, 10432, 10668, 10824])

      oenv = librosa.onset.onset_strength(y=self.audio, sr=self.sr, hop_length=128)
      tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=self.sr, hop_length=128)
  
      algo = rpt.KernelCPD(kernel="linear").fit(tempogram.T)
  
      n_bkps = 5
      bkps = algo.predict(n_bkps=n_bkps)
      bkps_times = librosa.frames_to_time(bkps, sr=self.sr, hop_length=128)
      if bkps_times[-1] < self.duration:
        bkps_times = np.append(bkps_times, self.duration)
  
      temp_segments = (self.num_frames*bkps_times/bkps_times[-1]).astype(np.int32)
      temp_segments[-1] = temp_segments[-1]-1 # make num_frames to num_frames-1
  
      if temp_segments[0] > temp_segments[1]/2:
        temp_segments = np.insert(temp_segments, 0, 0)
        return temp_segments
      else:
        temp_segments[0] = 0
        return temp_segments


  def get_peaks(self, signal, distance=25, prominence=0.2, width=5, height=0.5):

      """
      Get the mountain peak location of a signal.
      
      """

      peak_idx, _ = scipy.signal.find_peaks(signal, distance=distance, prominence=prominence, width=width, height=height)
      return peak_idx
      
  def halfgaussian_kernel1d(self, sigma=10, radius=50):
      
      """
      Creates a 1-D Half-Gaussian convolution kernel.
      
      """
      sigma2 = sigma * sigma
      x = np.arange(0, radius+1)
      phi_x = np.exp(-0.5 / sigma2 * x ** 2)
      phi_x = phi_x / phi_x.sum()

      return phi_x

  def get_smoothed_peaks(self, signal, radius=50, distance=25, prominence=0.2, width=5, height=0.5, reverse=False, just_impulse=False):

      """
      Get the indices of the peaks in the signal.
      Using the indices make an impulse at the indices with magnitude given by original signal at that index.
      Using a half guassian kernel, convolve impulses to create half gaussians (right side of gaussian) at each impulse.
      Shift the signal so that the start of the half gaussian occurs at the index of the impulse (due to convolution, the middle of the half gauss occurs at the index of the impulse).
      
      If I only want the impulse signal that can be returned.
      If I want to reverse the half gaussian that can also be returned (left side of gaussian).
      
      """
      peaks_idx = self.get_peaks(signal, distance=25, prominence=0.2, width=5, height=0.5)

      impulse_signal = np.zeros_like(signal)
      impulse_signal[peaks_idx] = signal[peaks_idx]
      if just_impulse:
        return impulse_signal

      hgauss = self.halfgaussian_kernel1d(radius=radius)
      if reverse:
        hgauss = hgauss[::-1]

      return np.pad(scipy.ndimage.convolve1d(impulse_signal, hgauss/hgauss.max(), axis=-1), (radius // 2, 0), mode='constant')[:-(radius // 2)]

  def make_plateu(self, array):
      
      """
      Compress the height of the nonzeros in the signal and shift upwards.
      Turn zeros to ones so that signal is now a uniform one with dips at the indices of the impulses.
    
      """
    
      new_array = array.copy()
      new_array[new_array != 0 ] = (new_array[new_array != 0 ]/5) + 0.8
      new_array[new_array==0] = 1
      return new_array

  def setup(self, reverse, radius_1, radius_2):
    
      """
      Setup the desired signals to use.
      
      """
      
      if isinstance(self.melody, (np.ndarray)):
          self.melody_chroma = self.get_chroma(self.melody, self.sr, self.num_frames, use_harm=True) # (n, 12) 
      
      if isinstance(self.drums, (np.ndarray)):
      
          self.drums_onsetsHigh = self.get_onsets(self.drums, self.sr, self.num_frames, fmin=4000, smooth=4, power=1.2) # shape: (n,) scaled btwn 0-1
          self.drums_onsetsMid = self.get_onsets(self.drums, self.sr, self.num_frames, fmin=500, fmax=2000, smooth=4, power=1.2) # shape: (n,) scaled btwn 0-1
          self.drums_onsetsLow = self.get_onsets(self.drums, self.sr, self.num_frames, fmax=150, smooth=4, power=1.2) # shape: (n,) scaled btwn 0-1
      
          self.drums_onsetsHigh_2 = self.get_smoothed_peaks(self.drums_onsetsHigh, radius=radius_1, height=0.5)
          self.drums_onsetsMid_2 = self.get_smoothed_peaks(self.drums_onsetsMid, radius=radius_1, height=0.5)
          self.drums_onsetsLow_2 = self.get_smoothed_peaks(self.drums_onsetsLow, radius=radius_1, height=0.5)
      
          self.drums_onsetsHigh_peaks = self.get_smoothed_peaks(self.drums_onsetsHigh, radius=radius_1, just_impulse=True, height=0.5, distance=20, width=4, prominence=0.15)
          self.drums_onsetsMid_peaks = self.get_smoothed_peaks(self.drums_onsetsMid, radius=radius_1, just_impulse=True, height=0.5, distance=20, width=4, prominence=0.15)
          self.drums_onsetsLow_peaks = self.get_smoothed_peaks(self.drums_onsetsLow, radius=radius_1, just_impulse=True, height=0.5, distance=20, width=4, prominence=0.15)
      
      if isinstance(self.bass, (np.ndarray)):
      
          self.bass_onsetsHigh = self.get_onsets(self.bass, self.sr, self.num_frames, fmin=2000, smooth=4) # shape: (n,) scaled btwn 0-1
          self.bass_onsetsHigh_2 = self.get_smoothed_peaks(self.bass_onsetsHigh, radius=radius_1, height=0.5)
          self.bass_onsetsHigh_2_r = self.get_smoothed_peaks(self.bass_onsetsHigh, radius=radius_2, reverse=True, height=0.5)
          self.bass_onsetsHigh_2_r_p = scipy.ndimage.gaussian_filter1d(self.make_plateu(self.bass_onsetsHigh_2_r), 0.5)
          self.bass_onsetsHigh_peaks = self.get_smoothed_peaks(self.bass_onsetsHigh, radius=radius_1, just_impulse=True, height=0.5, distance=20, width=4, prominence=0.15)
      
      if isinstance(self.audio, (np.ndarray)):
      
          self.tempo_segments = self.tempo_segmentation()
      
      if reverse and isinstance(self.drums, (np.ndarray)):
      
          self.drums_onsetsHigh_2_r = self.get_smoothed_peaks(self.drums_onsetsHigh, radius=radius_2, reverse=True, height=0.5)
          self.drums_onsetsMid_2_r = self.get_smoothed_peaks(self.drums_onsetsMid, radius=radius_2, reverse=True, height=0.5)
          self.drums_onsetsLow_2_r = self.get_smoothed_peaks(self.drums_onsetsLow, radius=radius_2, reverse=True, height=0.5)
          self.drums_onsetsHigh_2_r_p = scipy.ndimage.gaussian_filter1d(self.make_plateu(self.drums_onsetsHigh_2_r), 0.5)
          self.drums_onsetsMid_2_r_p = scipy.ndimage.gaussian_filter1d(self.make_plateu(self.drums_onsetsMid_2_r), 0.5)
          self.drums_onsetsLow_2_r_p = scipy.ndimage.gaussian_filter1d(self.make_plateu(self.drums_onsetsLow_2_r), 0.5)


def get_noise(num_frames, log2_resolution=9):

    """
    Generates both the "noisy" noise and "smooth" noise to use.
    
    Gaussian filter the image both normally and spatially (across the sequence of images).
      Why?
        Gauss the image to remove too much noise.
        Gauss the spatial dimension so that there is smooth transition between frames.
    I omit the last resolution as it took too much memory.
    
    Output: List of length ((log2_resolution-1) - 2) of tuples (noise, gauss_noise) where gauss_noise and noise has shape (n, res, res, 1).
    
    """
  
    noise_list = []
    sigma = 1
  
    noise_img_sigma = 0.2
    gauss_img_sigma = 0.6
    
    noise_spatial_sigma = 7
    gauss_spatial_sigma = 15
    
    for res in tqdm(range(2, log2_resolution)):
        # Make noises
        noise = np.random.normal(0, sigma, size = (num_frames, 2**res, 2**res, 1))
        gauss_noise = np.random.normal(0, sigma, size = (num_frames, 2**res, 2**res, 1))
        
        # Filter the image
        noise_filt_img = scipy.ndimage.gaussian_filter(noise, noise_img_sigma) # (n, res, res, 1)
        # Filter spatially
        noise_reshape = noise_filt_img.reshape((num_frames, (2**res)*(2**res)))
        noise_reshaped_gauss = scipy.ndimage.gaussian_filter1d(noise_reshape, sigma=noise_spatial_sigma, axis=0)
        noise = noise_reshaped_gauss.reshape((num_frames, 2**res, 2**res, 1))
        
        # Filter the image
        gauss_noise_filt_img = scipy.ndimage.gaussian_filter(gauss_noise, gauss_img_sigma) # (n, res, res, 1)
        # Filter spatially
        gauss_noise_reshape = gauss_noise_filt_img.reshape((num_frames, (2**res)*(2**res)))
        gauss_noise_reshaped_gauss = scipy.ndimage.gaussian_filter1d(gauss_noise_reshape, sigma=gauss_spatial_sigma, axis=0)
        gauss_noise = gauss_noise_reshaped_gauss.reshape((num_frames, 2**res, 2**res, 1))
        # Append Noises
        noise_list.append((noise, gauss_noise))
  
    return noise_list

