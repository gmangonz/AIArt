# AIART
This repo contains the code for my Film 499 visual art project seen [here](https://youtu.be/CnGdIXuR7QU).

UPDATE: I have made significant improvements to the AudioReactivity, the old README.md for reference is located in /src.

## Usage

[AIART.py](https://github.com/gmangonz/Personal/blob/main/AIART/AIART.ipynb) gives the process of how to train StyleGAN2 or run ```AudioReactive```, just make sure to change Parameters.py.

## How it works

To learn how StyleGAN2 works, it'll be best to look at the resources provided in References as those are the what helped me learn to implement this project from scratch. To get AudioReactive to work, I basically have to get the 3 inputs needed for StyleGAN2 that have dependencies on music signals. However one of the inputs is a 4x4 constant image, so I only have to get 2 inputs. These are obtained with ```.get_latents(...)``` and ```.get_noise(...)``` within ```AudioReacive```.

Now, what aspects of the music do I use? As a reference, some of these signals I extract from the audio are shown below.

- `self.audio_chroma`:
<img src="./images/audio_decompose.jpg" width=500px>

- `self.drums_rms`:
<img src="./images/drums_rms.jpg" width=500px>

- `rms_signal_for_noise`:
<img src="./images/drums_rms_signal_for_noise.jpg" width=500px>

- `sawtooth_signal`:
<img src="./images/sawtooth.jpg" width=500px>

- drums onsets smoothed peaks (Used for network bending):
<img src="./images/drums_onsets_smoothed_peaks.jpg" width=500px>

- drums onsets smoothed peaks plateud version (Used only for Zoom for network bending):
<img src="./images/drums_onsets_smoothed_peaks_plateu.jpg" width=500px>

To set up the latents, I have to get them to a Tensor of shape (1, 8, 512) for the 8 blocks in my version of StyleGAN2. However that is to only make one image, so I actually have to get an array of (num_frames, 8, 512) where num_frames = duration * fps. The latents are dependent on 2 things: Tempo Segmentation and Chromagram/Decomposition of the audio. The tempo segmentation divides the audio into segments where the tempo changes. Chromagram/Decomposition of the audio produces 12 components which can be used to create weighted latents. 

### Tempo Segmentation

After running `tempo_segmentation()` within `Music_Processing.py`, I have different points indicating changes in tempo. These are seen by the yellow points in the following image.
<img src="./images/tempo_segmentation.jpg" width=500px>

Now, to make an array of (num_frames, 8, 512) I actually need to create an array of (num_frames, 1, 512). I do this by assigning each segment with a unique latent vector. However to avoid abrupt changes, I interpolate between latents whenever there is a change within ```.get_latents_from_impulse()``` using ```slerp()```. 

### Decomposition
Passing the audio into `decompose()` I obtain 12 compenents of the audio (this is significantly faster than `get_chroma()` and yields similar effects) which can be best explained [here](https://librosa.org/doc/main/generated/librosa.decompose.decompose.html). This gives an array of shape (num_frams, 12). If we take an array of 12 latent vectors i.e (12, 1, 512), reshape it to (12, 512), and perform matrix multiplication and reshape, we get (num_frams, 1, 512) which creates a sequence of weighted latents. 


### Form latents
From Tempo Segmentation, we have (num_frames, 1, 512) and from Decomposition, we have (num_frams, 1, 512). However, the input required is (num_frames, 8, 512). Therefore, I tile each and stack them on top of each other to create latent vectors of shape (num_frames, 8, 512). How to tile and which to stack on top of which leads to different effects, however, as preference I like tiling the tempo segmentation output to (num_frames, 6, 512) and the decomposition to (num_frames, 2, 512) and stack the tempo segmentation output on top of the decomposition output. 

### Latent modulation




## TODO:

- [x] Tempo Segmentation fix.
- [x] Network Bending - Translation and Rotation fix.
- [x] Network Bending - Add more transformations.
- [ ] Research real time music component seperation.
- [x] Make ```self.get_noise()``` run in real time.
- [x] Create 3D faux noise.
- [x] Play with displacement maps and `tfa.image.dense_image_warp`
- [ ] Create my own dataset.
- [ ] Train model longer to get good images with any seed.
- [ ] StyleGAN3.
- [ ] 1024x1024 images.

# References

- https://github.com/dvschultz/ml-art-colabs
    - Inspired me to combine DL with Art
- https://github.com/christianversloot/machine-learning-articles/blob/main/stylegan-a-step-by-step-introduction.md
    - Where I first learned how StyleGAN works 
- https://cv-tricks.com/how-to/understanding-stylegan-for-image-generation-using-deep-learning/
    - Another StyleGAN resource
- https://github.com/beresandras/gan-flavours-keras
    - Helped in some aspects of my implementation of StyleGAN2
- https://github.com/NVlabs/stylegan2
    - Official StyleGAN2 implementation
- https://arxiv.org/pdf/1912.04958.pdf
    - Official StyleGAN2 paper
- https://nn.labml.ai/gan/stylegan/index.html
    - Helped me implement StyleGAN2
- https://github.com/JCBrouwer/maua-stylegan2
    - Inspired this project