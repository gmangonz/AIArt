import numpy as np
from scipy import signal
import librosa.display
import scipy
import tensorflow as tf
from tqdm.autonotebook import tqdm
import pandas as pd
from functools import reduce
from functools import partial

from Parameters import functions_mapping, func_params, audio_config, signals_config
from Network_Bending import DisplacementMap
from Music_Processing import tempo_segmentation


def composite_function(*funcs):

    """
    Input:
      funcs: a tuple of tuples that contain function name and function kwags

    """
    funcs = list(funcs)
    funcs = list(map(lambda x: partial( functions_mapping.get(x[0]), **(func_params.get(x[1]) if isinstance(x[1], str) else x[1])), funcs))
    funcs = tuple(funcs)

    def compose(f_in, g_in):
        return lambda x : g_in(f_in(x))
    return reduce(compose, funcs)


class Config(dict):

  def __init__(self, dictionary):

    for k, v in dictionary.items():
      if isinstance(v, dict):
        v = dict([(f'{k}_{in_k}', in_v) for in_k, in_v in v.items()])
        self.__init__(v)

      else:
        setattr(self, k, v)

  def update(self, dictionary):
    for k, v in dictionary.items():
      if isinstance(v, dict):
        v = dict([(f'{k}_{in_k}', in_v) for in_k, in_v in v.items()])
        self.update(v)
      else:
        setattr(self, k, v)


class MusicProcessing(Config):

  def __init__(self, dict_config):
    super(MusicProcessing, self).update(dict_config)

  def get_audio_signal(self, audio, funcs, kwargs):

        audio_to_use = audio_config.get(audio)
        composite_func = composite_function( *tuple(zip(funcs, kwargs)) )
        return composite_func(audio_to_use)

  def make_transformation_dict(self, name, bending_func, audio, funcs, kwargs):

      if isinstance(bending_func, DisplacementMap):
        self.transformation_dict[name] = [bending_func, None]
      else:
        self.transformation_dict[name] = [bending_func, self.get_audio_signal(audio, funcs, kwargs)]

  def get_tranformations_from_config(self):

    self.transformation_dict = {}
    for name, bending_func, audio, funcs, kwargs in zip(self.transformation_dict_names, self.transformation_dict_bending, self.transformation_dict_audio, self.transformation_dict_funcs, self.transformation_dict_kwargs):
        self.make_transformation_dict(name, bending_func, audio, funcs, kwargs)

  def get_noise_signals_from_config(self):

    for i, (name, funcs, kwargs) in enumerate(zip(self.noise_audio, self.noise_funcs, self.noise_kwargs)):
      setattr(self, f'noise_{i}', self.get_audio_signal(name, funcs, kwargs))

  def get_chroma_signals_from_config(self):

    for i, (name, funcs, kwargs) in enumerate(zip(self.chromagram_audio, self.chromagram_funcs, self.chromagram_kwargs)):
      setattr(self, f'{name}_chroma', self.get_audio_signal(name, funcs, kwargs))

  def get_onsets_signals_from_config(self):

    for i, (name, funcs, kwargs) in enumerate(zip(self.onsets_audio, self.onsets_funcs, self.onsets_kwargs)):
      setattr(self, f'{name}_{funcs[0]}', self.get_audio_signal(name, funcs, kwargs))

  def fix_impulse(self, impulse_idx):

    if impulse_idx[0] < self.num_frames*0.01:
        impulse_idx[0] = 0
    if impulse_idx[0] != 0:
        impulse_idx = np.insert(impulse_idx, 0, 0)
    if impulse_idx[-1] > self.num_frames * 0.99:
        impulse_idx[-1] = self.num_frames - 1
    if impulse_idx[-1] >= self.num_frames:
        impulse_idx[-1] = self.num_frames - 1
    if impulse_idx[-1] != self.num_frames-1:
        impulse_idx = np.append(impulse_idx, self.num_frames-1)

    return impulse_idx

  def get_impulse_signals(self, params):

    self.segmentation_idxs = self.fix_impulse(tempo_segmentation(self.audio, self.num_frames))

  def setup(self, params):

    print("Getting chromagrams...")
    self.get_chroma_signals_from_config()

    print("Getting onsets...")
    self.get_onsets_signals_from_config()

    print("Getting transformations dict...")
    self.get_tranformations_from_config()

    # print("Getting noise signals...")
    # self.get_noise_signals_from_config()

    print("Getting impulse signals...")
    self.get_impulse_signals(params)