import random
import time

import cv2
import kornia.augmentation as K
import matplotlib.pyplot as plt
import torch
from kornia.constants import Resample, SamplePadding

from discriminator import Discriminator
from generator import Generator
from mapping_network import Mapping


def hard_step(values: torch.Tensor) -> torch.Tensor:
    """
    A hard sigmoid function, useful for binary accuracy calculation from logits.
    Negative values -> 0.0, positive values -> 1.0.

    Parameters
    ----------
    values : torch.Tensor
        Input tensor

    Returns
    -------
    torch.Tensor
        Binary values
    """
    return 0.5 * (1.0 + torch.sign(values))


class RandomChoice(torch.nn.Module):
    def __init__(self, transforms, probabilities=None):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)
        if probabilities is None:
            self.probabilities = [1 / len(transforms)] * len(transforms)
        else:
            assert len(transforms) == len(probabilities), "Mismatch between transforms and probabilities"
            assert abs(sum(probabilities) - 1.0) < 1e-6, "Probabilities must sum to 1"
            self.probabilities = probabilities

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        transform = random.choices(self.transforms, weights=self.probabilities, k=1)[0]
        return transform(images)


class AdaptiveAugmenter(torch.nn.Module):
    def __init__(
        self,
        max_translation: float = 0.4,
        max_zoom: float = 0.4,
        max_rotation: int = 90,
        init_probability: float = 0.0,
        **kwargs,
    ):
        super(AdaptiveAugmenter, self).__init__(**kwargs)

        self.probability = torch.nn.Parameter(torch.as_tensor(init_probability), requires_grad=False)
        self.target_accuracy = 0.8  # 0.85, 0.95
        self.integration_steps = 1000  # 1000, 2000

        self._interpolation = Resample.NEAREST
        self._mode = SamplePadding.REFLECTION

        self.augmenter = torch.nn.Sequential(
            # Horizontal Flip
            K.RandomHorizontalFlip(p=0.8),
            # # Random Geometric Transformations
            RandomChoice(
                [  # Random Translation
                    K.RandomAffine(
                        degrees=0,
                        translate=(max_translation, max_translation),
                        resample=self._interpolation,
                        padding_mode=self._mode,
                        p=0.8,
                    ),
                    # Random Rotation
                    K.RandomAffine(
                        degrees=max_rotation,
                        resample=self._interpolation,
                        padding_mode=self._mode,
                        p=0.8,
                    ),
                    # Random Zoom
                    K.RandomAffine(
                        scale=(1 - max_zoom, 1),
                        degrees=0,
                        resample=self._interpolation,
                        padding_mode=self._mode,
                        p=0.8,
                    ),
                ],
                probabilities=[1 / 3] * 3,
            ),
            # # Random Distortions
            RandomChoice(
                [  # Random Elastic
                    K.RandomElasticTransform(
                        kernel_size=(41, 41),
                        sigma=(24.0, 24.0),
                        alpha=(0.6, 1.6),
                        resample=self._interpolation,
                        padding_mode=self._mode.name.lower(),
                        p=0.6,
                    ),
                    # Random Perspective
                    K.RandomPerspective(distortion_scale=0.3, resample=self._interpolation, p=0.6),
                ],
                probabilities=[0.5, 0.5],
            ),
            # Extra Augmentations
            RandomChoice(
                [
                    K.RandomContrast(contrast=(0.2, 0.6), p=0.7),
                    K.RandomPlasmaBrightness(roughness=(0.4, 0.8), intensity=(0.2, 0.8), p=0.7),
                    K.RandomHue(hue=(0, 0.5), p=0.7),
                    K.RandomSaturation(saturation=(0.3, 1.7), p=0.7),
                ],
                probabilities=[0.25, 0.25, 0.25, 0.25],
            ),
            # Add Noise
            K.RandomGaussianNoise(mean=0.0, std=0.4, p=0.4),
        )

    def forward(self, images: torch.Tensor):
        # [B, C, H, W]
        if self.training:
            B = images.shape[0]
            augmented = self.augmenter(images)
            augmentation_values = torch.rand(size=(B, 1, 1, 1), requires_grad=True)  # Generate [0-1] random numbers
            images = torch.where(augmentation_values.to(images.device) <= self.probability, augmented, images)
            images = torch.clip(images, 0.0, 1.0)
        return images

    def update(self, real_logits):

        current_accuracy = torch.mean(hard_step(real_logits))
        # the augmentation probability is updated based on the dicriminator's accuracy on real images
        accuracy_error = current_accuracy - self.target_accuracy
        self.probability = torch.clip(self.probability + accuracy_error / self.integration_steps, 0.0, 1.0)  # FIXME?


class StyleGAN2(torch.nn.Module):
    def __init__(
        self,
        batch_size: int,
        log2_resolution: int,
        latent_dim: int,
        min_num_features: int = 32,
        max_num_features: int = 512,
        beta: float = 0.99,
        **kwargs,
    ):
        super(StyleGAN2, self).__init__(**kwargs)

        self.batch_size = batch_size
        # Model Parameters
        self.log2_end_res = log2_resolution
        self.num_blocks = log2_resolution - 1  # Because we are forcing it to start at 4 (or 2 in log2)
        self.latent_dim = latent_dim
        self.max_num_features = max_num_features

        # Setup Discriminator
        self.discriminator = Discriminator(
            max_log2_res=log2_resolution,
            max_num_features=max_num_features,
            min_num_features=int(min_num_features * 2),
        )

        # Setup Generator
        self.generator = Generator(
            log2_resolution,
            latent_dim=latent_dim,
            max_num_features=max_num_features,
            min_num_features=min_num_features,
        )

        # Setup Mapping
        self.mapping_network = Mapping(num_blocks=self.num_blocks, latent_dim=latent_dim, num_layers=8)

        # Setup ADA
        self.ada = AdaptiveAugmenter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train_forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:

        B = x.shape[0]
        z = torch.randn((B, self.latent_dim), device=x.device)
        const_input = torch.ones((B,) + (self.max_num_features, 4, 4), device=x.device)
        noise = [torch.randn((2, B, 1, 2**res, 2**res), device=x.device) for res in range(2, self.log2_end_res + 1)]

        w_mapping = self.mapping_network(z)
        generated_images = self.generator(const_input, w_mapping, noise)
        generated_images = self.ada(generated_images)

        discriminate_real = self.discriminator(x)
        discriminate_fake = self.discriminator(generated_images)

        return {
            "generated_images": generated_images,
            "discriminate_real": discriminate_real,
            "discriminate_fake": discriminate_fake,
            "const_input": const_input,
        }

    def forward(self, x: torch.Tensor | None = None, N: int = 1) -> dict[str, torch.Tensor] | torch.Tensor:
        if self.training:
            assert x is not None
            return self._train_forward(x)

        const_input = torch.ones((N,) + (self.max_num_features, 4, 4), device=self.device)
        w_mapping = self.mapping_network(z=torch.randn((N, self.latent_dim), device=self.device))
        noise = [torch.randn((2, N, 1, 2**res, 2**res), device=self.device) for res in range(2, self.log2_end_res + 1)]
        return self.generator(x=const_input, w=w_mapping, noise=noise)


if __name__ == "__main__":

    # Run StyleGAN2
    log2_resolution = 9
    model = StyleGAN2(batch_size=None, log2_resolution=log2_resolution, latent_dim=512)
    model.to(torch.device("cuda"))

    image = cv2.imread("images/sample.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (2**log2_resolution, 2**log2_resolution))
    image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)  # [H, W, C] -> [B, H, W, C] -> [B, C, H, W]
    image = image.repeat(4, 1, 1, 1)
    image = (image.float() / 255).to(torch.device("cuda"))

    model.train()
    out = model(image)

    # Visualize AdaptiveAugmenter
    batchsize = 36
    augmenter = AdaptiveAugmenter(init_probability=0.5)
    augmenter.to(torch.device("cuda"))

    image = cv2.imread("images/sample.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))

    start = time.time()
    image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)  # [H, W, C] -> [B, H, W, C] -> [B, C, H, W]
    image = (image.float() / 255).to(torch.device("cuda"))
    augmented_image = augmenter(image.repeat((batchsize, 1, 1, 1)))
    augmented_image = augmented_image.permute(0, 2, 3, 1).cpu().numpy()  # [B, C, H, W] -> [B, H, W, C]
    print(f"::TIME TAKE - {time.time() - start}::")

    rows = 6
    cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, img in enumerate(augmented_image):
        ax = axes[i]
        ax.imshow(img)  # Show the i-th image
        ax.axis("off")

    plt.tight_layout()
    plt.show()
