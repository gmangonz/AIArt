import os
from easydict import EasyDict
import librosa
import scipy
from mutagen.mp3 import MP3
import tensorflow as tf
import numpy as np

from Music_Processing import get_onsets, get_smoothed_peaks, make_plateu, get_chroma, calculate_rms, normalize, get_beat_plp, decompose, calculate_ema
from Network_Bending import Zoom

SHORT_AUDIO = False

params = EasyDict()
params.log2res = 9
params.BATCH_SIZE = 32
params.steps = 200000 # 100, 400, 2000, 4000, 12000, 100000, 200000
params.max_features = 512
params.duration = 90 # in seconds
params.fps = 24
params.tempo_latents_seed = 8
params.drum_latents_seed = 42

directory_params = EasyDict()
directory_params.base_directory = r'/content/drive/MyDrive/Film 499/'
directory_params.music_directory = r'/content/drive/MyDrive/DL-CV-ML Projects/AIART/audio/Short_Final' if SHORT_AUDIO else r'/content/drive/MyDrive/DL-CV-ML Projects/AIART/audio' # /Short_Final'
directory_params.training_checkpoint_directory = r'/content/drive/MyDrive/DL-CV-ML Projects/AIART/training_checkpoint/'
directory_params.ckpt_folder_name = '4_17_22/Final_Test/'
directory_params.ckpt_name = 'ckpt'
directory_params.output_dir = '/content/drive/MyDrive/DL-CV-ML Projects/AIART/outputs/'
directory_params.output_dir_name = '4_19/I/'
directory_params.dataset_folder = '/content/drive/MyDrive/DL-CV-ML Projects/All_Data/AIART/' # 'DataSet/Abstract_Imgs/train/'
directory_params.audio_name = 'test.mp3' if SHORT_AUDIO else 'Music.mp3' # test, Music
directory_params.melody_name = 'other.wav'
directory_params.drums_name = 'drums.wav'
directory_params.bass_name = 'bass.wav'
directory_params.vox_name = 'vocals.wav'


img_size = 2**params.log2res
ds_directory = os.path.join(directory_params.base_directory, directory_params.dataset_folder)
audiofile = os.path.join(directory_params.music_directory, directory_params.audio_name)
melody_audiofile = os.path.join(directory_params.music_directory, directory_params.melody_name)
drums_audiofile = os.path.join(directory_params.music_directory, directory_params.drums_name)
bass_audiofile = os.path.join(directory_params.music_directory, directory_params.bass_name)
vox_audiofile = os.path.join(directory_params.music_directory, directory_params.vox_name)


audio = MP3(audiofile)
params.duration = audio.info.length


audio, sr_audio = librosa.load(audiofile, offset=0, duration=params.duration)
melody, sr_melody = librosa.load(melody_audiofile, offset=0, duration=params.duration)
drums, sr_drums = librosa.load(drums_audiofile, offset=0, duration=params.duration)
bass, sr_bass = librosa.load(bass_audiofile, offset=0, duration=params.duration)
vox, sr_vox = librosa.load(vox_audiofile, offset=0, duration=params.duration)

num_frames = int(round(params.duration*params.fps))
shape = (512, 512)

base_resolutions = {# Min, Max, Max amplitude
        (4, 4):     (2, 3 , 2),
        (8, 8):     (2, 5 , 3),
        (16, 16):   (3, 5 , 4),
        (32, 32):   (3, 8 , 6),
        (64, 64):   (4, 12, 10),
        (128, 128): (5, 16, 16),
        (256, 256): (6, 24, 32),
        (512, 512): (7, 32, 64)
}


radius_1 = 36
radius_2 = 2

functions_mapping = {
    'onset':     get_onsets,
    'chroma':    get_chroma,
    's_peaks':   get_smoothed_peaks,
    'plateu':    make_plateu,
    'gauss':     scipy.ndimage.gaussian_filter1d,
    'rms':       calculate_rms,
    'norm':      normalize,
    'ema':       calculate_ema,
    'beat_plp':  get_beat_plp,
    'decompose': decompose,
    'resample':  scipy.signal.resample,
    'cumsum':    np.cumsum,
    'clip':      np.clip
}


func_params = {
    'onsets_mid':            dict(sr=sr_audio, n_frames=num_frames, fmin=500, fmax=2000, smooth=4, power=1.2),
    'onsets_mid_2':          dict(sr=sr_audio, n_frames=num_frames, fmin=0, fmax=2200, smooth=1, power=1),
    'bass_high':             dict(sr=sr_audio, n_frames=num_frames, fmin=1000, smooth=4), # 2000
    'smoothed_peaks_a':      dict(radius=6, distance=20, prominence=0.6, width=2, height=0.1, kernel_type='full'),
    'smoothed_peaks_b':      dict(radius=4, distance=20, prominence=0.6, width=2, height=0.1, kernel_type='full', reverse=True), # radius=radius_2, kernel_type='half', reverse=True
    'smoothed_peaks_c':      dict(radius=3,                                       height=0.3, kernel_type='full', reverse=True), # height=0.5, kernel_type='half'
    'decompose':             dict(n_frames=num_frames, smooth=3, margin=5),
    'chroma':                dict(sr=sr_audio, n_frames=num_frames, smooth=3, hop_length=512, use_harm=True),
}

zoom_params = {
    'generator__block'  : (1, 0.80), # 4x4 -> 8x8, network bending will occur on 8x8 image
    'generator__block_1': (1, 0.80), # 16x16
    'generator__block_2': (1, 0.85), # 32x32
    'generator__block_3': (1, 0.85), # 64x64
    'generator__block_4': (1, 0.90), # 128x128
    'generator__block_5': (1, 0.90), # 256x256
    'generator__block_6': (1, 0.95), # 512x512
}


audio_config = {
    'audio':      audio,
    'melody':     melody,
    'drums':      drums,
    'bass':       bass,
    'num_frames': num_frames,
    'sr':         sr_audio,
    'melody_latents_seeds': [388, 51, 97, 251, 389, 447, 185, 35, 166, 8, 21, 39], # 12
    'drums_latents_seeds' : [42, 49, 100, 430, 122, 335, 664, 12, 123, 696, 465, 467],
    'bass_latents_seeds'  : [985, 267, 619, 388, 266, 776, 875, 643, 768, 636, 643, 210],
}


signals_config = {
    'transformation_dict_names'  : ['generator__block_3'                , 'generator__block_5'],
    'transformation_dict_bending': [[Zoom()]                            , [Zoom()]],
    'transformation_dict_audio'  : ['drums'                             , 'bass'],
    'transformation_dict_funcs'  : [['onset'       , 's_peaks'         ], ['onset'    , 's_peaks'         ]],
    'transformation_dict_kwargs' : [['onsets_mid_2', 'smoothed_peaks_b'], ['bass_high', 'smoothed_peaks_c']],

    'chromagram_audio' : ['melody'],   #, 'melody'  , 'audio'
    'chromagram_funcs' : ['chroma'], #, ['chroma'], ['decompose']
    'chromagram_kwargs': ['chroma'], #, ['chroma'], ['decompose']

    'onsets_audio' : ['drums'],
    'onsets_funcs' : [['rms'                 , 'norm'        , 'resample'          , 'clip']],
    'onsets_kwargs': [[dict(window_size=None), dict(a=0, b=1), dict(num=num_frames), dict(a_min=0, a_max=1)]],
}