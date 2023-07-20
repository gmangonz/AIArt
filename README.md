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

## TODO:

- [x] Tempo Segmentation fix.
- [x] Network Bending - Translation and Rotation fix.
- [x] Network Bending - Add more transformations.
- [ ] Research real time music component seperation.
- [x] Make ```self.get_noise()``` run in real time.
- [x] Create 3D faux noise.
- [x] Play with displacement maps, `tfa.image.dense_image_warp`
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