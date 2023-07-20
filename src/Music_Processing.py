import numpy as np
import scipy
import librosa
from scipy import signal
import tensorflow as tf
import pandas as pd

def normalize(signal, a, b):

    """ Normalizes input to values between a and b """

    A = np.min(signal)
    B = np.max(signal)

    C = (a-b + 1e-6)/(A-B + 1e-6)
    k = (C*A - a)/(C)
    return (signal-k)*C

def get_beat_impulse_idx(audio, sr, num_frames, params):

    """ Get indices of the beats in the audio. Returns np.array of indices. """

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)

    beats_idx = np.round(librosa.frames_to_time(beats, sr=sr)*params.fps).astype(np.int32)
    return beats_idx

def get_beat_plp(drums, sr, margin=3, hop_length=256, win_length=384, lognorm=True):

    """
    Uses beat.plp from librosa to get information on the beat track from the drums.
    Normalizes the signal returned by beat.plp
    """

    audio_percussive = librosa.effects.percussive(y=drums, margin=margin)
    onset_env = librosa.onset.onset_strength(y=audio_percussive, sr=sr)
    if lognorm:
        prior = scipy.stats.lognorm(loc=np.log(120), scale=120, s=0.5)
        return librosa.util.normalize(librosa.beat.plp(onset_envelope=onset_env, hop_length=hop_length, win_length=win_length, sr=sr, prior=prior))
    return librosa.util.normalize(librosa.beat.plp(onset_envelope=onset_env, hop_length=hop_length, win_length=win_length, sr=sr))

def get_onsets(audio, sr, n_frames, margin=3, power=1, fmin=20, fmax=8000, smooth=1, hop_length=128):

    """
    Gets the onset of the percussive aspects of given audio; detects changes in audio.
    Resamples the signal to the desired frames where n_frames = duration*fps.
    Clips the resample to orignal min and max.
    Gaussian filter the signal to reduce noise.
    Normalize.

    """
    y_perc = librosa.effects.percussive(y=audio, margin=margin)
    onset = librosa.onset.onset_strength(y=y_perc, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length)
    onset = np.clip(signal.resample(onset, n_frames), onset.min(), onset.max())
    onset = scipy.ndimage.gaussian_filter1d(onset, smooth, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    onset = (onset - np.min(onset))/np.ptp(onset)
    return onset ** power

def get_chroma(audio, sr, n_frames, margin=3, hop_length=512, use_harm=True, smooth=20):

    """
    Gets the chromagram of the harmonic aspects of given audio; expresses the audio's pitch content over the twelve chroma bands.
    Gaussian filter the rows.
    Resample the image to the desired frames where n_frames = duration*fps.
    Normalize so that rows adds to 1. Why rows? Shape is # (num_frames, 12), when we grab a frame we get (12,) and we want this array to add to 1.
    Sort the rows based on their sum, largest on top.

    """
    y_harm = librosa.effects.harmonic(y=audio, margin=margin) if use_harm == True else audio
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length) # (12, 10336)
    chroma = scipy.ndimage.gaussian_filter1d(chroma.T, smooth, axis=0, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0).T
    chroma = signal.resample(chroma.T, n_frames) # (num_frames, 12)

    chrom_class_norm = chroma / chroma.sum(axis = 1, keepdims = 1) # (num_frames, 12)
    return tf.cast(chrom_class_norm, tf.float32)

def decompose(audio, n_frames, smooth=6, margin=3):

    """
    Get the harmonic components of the audio.
    Get spectrogram.
    Decomposes the spectrogram into 12 components.
    Gaussian filter the rows.
    Transposes and normalizes so that rows adds to 1
    Resample the image to the desired frames where n_frames = duration*fps.
    Normalizes so that rows adds to 1 (Yes this is done twice. I tested this and found this returns best results)
    """

    audio_harmonic, _ = librosa.effects.hpss(audio, margin=margin)
    S = np.abs(librosa.stft(audio_harmonic))
    comps, acts = librosa.decompose.decompose(S, n_components=12, max_iter=500) # (12, n)
    acts = scipy.ndimage.gaussian_filter1d(acts, smooth, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0) # (12, n)
    acts = acts.T
    acts = acts / (acts.sum(axis = 1, keepdims = 1)+1e-9) # (num_frames, 12)
    acts = signal.resample(acts, n_frames)                # (num_frames, 12)
    return acts / (acts.sum(axis = 1, keepdims = 1)+1e-9) # (num_frames, 12)

def get_peaks(signal, distance=25, prominence=0.2, width=5, height=0.5):

    """
    Get the mountain peak location of a signal.

    """
    peak_idx, _ = scipy.signal.find_peaks(signal, distance=distance, prominence=prominence, width=width, height=height)
    return peak_idx

def halfgaussian_kernel1d(sigma=10, radius=50):

    """
    Creates a 1-D Half-Gaussian convolution kernel.

    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    return phi_x

def gaussian_kernel1d(sigma=10, radius=50):

    """
    Creates a 1-D Gaussian convolution kernel.

    """
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    return phi_x

def convolve(impulse_signal, gauss, radius):

    """ 
    Convolve the inpulse signal with gauss.

    Input:
        inpulse_signal: (num_frames,) with zeros except at certain indices where an inpulse lies.
        gauss: gaussian kernel for convolution

    Shift so that at each index where a impulse lied, the peak of convolution from the gaussian kernel is located at the same index. This is done with the following:
    pad the start due to the effects of convolution and return [:-(radius // 2)]

    """

    conv_out = scipy.ndimage.convolve1d(impulse_signal, gauss/gauss.max(), axis=-1)
    conv_pad = np.pad(conv_out, (radius // 2, 0), mode='constant')
    return conv_pad[:-(radius // 2)]

def get_smoothed_peaks(signal, radius=50, sigma=10, distance=25, prominence=0.2, width=5, height=0.5, kernel_type='half', reverse=True, just_impulse=False):

    """
    Get the indices of the peaks in the signal.
    Using the indices make an impulse at the indices with magnitude given by original signal at that index.
    Using a half guassian kernel, convolve impulses to create half gaussians (right side of gaussian) at each impulse.
    Shift the signal so that the start of the half gaussian occurs at the index of the impulse (due to convolution, the middle of the half gauss occurs at the index of the impulse).

    If I only want the impulse signal that can be returned.
    If I want to reverse the half gaussian that can also be returned (left side of gaussian).

    """

    assert kernel_type in ['half', 'full'], 'kernel should be one of the following: "reverse", "full"'
    peaks_idx = get_peaks(signal, distance=distance, prominence=prominence, width=width, height=height)

    impulse_signal = np.zeros_like(signal)
    impulse_signal[peaks_idx] = signal[peaks_idx]
    if just_impulse:
      return impulse_signal

    if kernel_type == 'full':
      kernel = gaussian_kernel1d(radius=radius, sigma=sigma)
    if kernel_type == 'half':
      kernel = halfgaussian_kernel1d(radius=radius, sigma=sigma)
      if reverse:
        kernel = kernel[::-1]

    return convolve(impulse_signal, kernel, radius)

def make_plateu(array, max_min = (1, 0.9)):

    """
    Compress the height of the nonzeros in the signal and shift upwards.
    Essentially turns zeros to ones so that signal is now a uniform one with dips at the indices of the impulses.

    """
    new_array = (1 - normalize(array, 1-max_min[0], 1-max_min[1])) #normalize(1-new_array[new_array != 0 ], 0.85, 1.0) #(new_array[new_array != 0 ]/5) + 0.8
    return new_array

def calculate_energy(signal, window_size=60, norm_with_mean=True):

    """ Batch the signal into windows of size window_size and calculate the energy of each batch and reassemble the signal, although the length will be smaller. """

    num = signal.shape[0] // window_size
    idx = window_size*num

    x = np.reshape(signal[:idx], (-1, window_size))
    if norm_with_mean:
        p = np.sum(x*x, axis=1)/(x.mean(axis=1)*x.size+1e-6)
        return p*x.size
    p = np.sum(x*x, axis=1)/(x.size+1e-6)
    return p*x.size

def butter_filt(cutoff, fs, ftype='low', order=3):

    """ Build Butter Filter. """

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype=ftype, analog=False)
    return b, a

def butter_filter(data, cutoff, fs, ftype, order=5):

    """ Filter with Butter Filter. """

    b, a = butter_filt(cutoff, fs, ftype, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def tempo_segmentation(audio, num_frames, cutoff=1, fs=100, ftype='low'):

    """ Cheap way to perform tempo segmentation. """

    rms         = calculate_rms(audio)
    filt_energy = butter_filter(rms, cutoff=cutoff, fs=fs, ftype=ftype)
    filt_energy = np.clip(scipy.signal.resample(filt_energy, num_frames), filt_energy.min(), filt_energy.max())
    return scipy.signal.find_peaks_cwt(1-filt_energy, np.arange(1,400))

def calculate_rms(audio, window_size=None):

    """ Calculates RMS of a signal. """

    return librosa.feature.rms(y=audio)[0]

def calculate_ema(audio):

    """ Runs Exponential moving average. """

    return pd.Series(audio).ewm(span = 2/0.05-1).mean()