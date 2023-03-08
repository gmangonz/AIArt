import os
from easydict import EasyDict

params = EasyDict()
params.log2res = 9
params.BATCH_SIZE = 32
params.steps = 200000 # 100, 400, 2000, 4000, 12000, 100000, 200000
params.max_features = 512
params.duration = 30 # in seconds
params.fps = 24

directory_params = EasyDict()
directory_params.base_directory = r'/content/drive/MyDrive/Film 499/'
directory_params.music_directory = r'/content/drive/MyDrive/Film 499/audio/Music_Final/'
directory_params.training_checkpoint_directory = r'/content/drive/MyDrive/Film 499/training_checkpoints/StyleGAN2/'
directory_params.ckpt_folder_name = '4_17_22/A/'
directory_params.ckpt_name = 'ckpt'
directory_params.output_dir = '/content/drive/MyDrive/Film 499/outputs/4_19/I/'
directory_params.dataset_folder = 'DataSet/Abstract_Imgs/train/'
directory_params.audio_name = 'Music.mp3'
directory_params.melody_name = 'other.wav'
directory_params.drums_name = 'drums.wav'
directory_params.bass_name = 'bass.wav'
directory_params.vox_name = 'vocals.wav'

