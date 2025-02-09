import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from StyleGAN2 import StyleGAN2, hard_step

__cwd__ = os.path.dirname(__file__)


class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.loss = 0.0
        self.losses = defaultdict(lambda: 0.0)
        self.metrics = defaultdict(lambda: 0.0)

    def update_loss(self, loss: torch.Tensor, **kwargs):
        """
        Update loss and accuracy values.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value
        """
        self.count += 1  # Count number of steps in epoch
        self.loss += loss.item()  # Single batch's loss

        for k, v in kwargs.items():
            self.losses[k] += v

    def update_metrics(self, **kwargs):
        """
        Update metric values.
        """
        for k, v in kwargs.items():
            self.metrics[k] += v

    def get_metrics(self) -> tuple[float, dict[str, float], dict[str, float]]:
        """
        Calculate average metrics on full epoch. Will divide total
        metrics by length of dataset.

        Returns
        -------
        tuple[float, float, dict[str, float]]
            Average loss, accuracy and extra metrics
        """
        avg_loss = self.loss / self.count
        for k, v in self.metrics.items():
            self.metrics[k] = v / self.count
        for k, v in self.losses.items():
            self.losses[k] = v / self.count
        return avg_loss, self.losses, self.metrics


class Trainer:
    def __init__(
        self,
        model: StyleGAN2,
        epochs: int,
        lr: float,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: torch.device = torch.device("cuda"),
        run_train: bool = True,
        run_val: bool = True,
        optimizer: dict[str, torch.optim.Optimizer] | None = None,
        scheduler: dict[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
        patience: int | None = None,
        output_path: str | None = None,
    ):
        self.device = device
        self.model = model.to(device)

        # Train loader
        self.run_train = run_train
        self.train_loader = train_loader

        # Val loader
        self.run_val = run_val
        if run_val:
            assert val_loader is not None, "Pass validation loader if running validation"
            self.val_loader = val_loader

        # Optimizer and scheduler
        self.optimizer = optimizer or {"model": optim.Adam(self.model.parameters(), lr=lr)}
        self.scheduler = scheduler

        self.__epochs = epochs
        self.__best_score = 0
        self.__early_stop = 0
        self.__patience = epochs or patience

        self.metrics = Metrics()
        self.output_path = output_path or __cwd__

    def _process_batch(
        self, step: int, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        raise NotImplementedError("Your subclass should implement this method!")

    def _train_loop(self, epoch: int) -> float:
        """
        Train the model on the training set.

        Parameters
        ----------
        epoch : int
            Current epoch

        Returns
        -------
        float
            Total average loss
        """
        self.model.train()
        self.metrics.reset()

        if epoch % 5 == 0:
            train_loop = tqdm.tqdm(
                self.train_loader,
                desc="Training",
                disable=False,
                leave=True,
                bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
            )
        else:
            train_loop = self.train_loader

        for step, batch in enumerate(train_loop):
            for _, opt in self.optimizer.items():
                opt.zero_grad()

            # Total loss, dict of respective losses, metrics
            loss, losses, metrics = self._process_batch(step, batch)

            loss.backward()
            for _, opt in self.optimizer.items():
                opt.step()

            self.metrics.update_loss(loss, **losses)
            self.metrics.update_metrics(**metrics)
            avg_loss, _, _ = self.metrics.get_metrics()

            if epoch % 5 == 0:
                train_loop.set_postfix({"loss": avg_loss})

        return avg_loss

    def _val_loop(self, epoch: int) -> float:
        """
        Run testing on the validation set.

        Parameters
        ----------
        epoch : int
            Current epoch

        Returns
        -------
        float
            Total average loss
        """
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():

            if epoch % 5 == 0:
                val_loop = tqdm.tqdm(
                    self.val_loader,
                    desc="Validation",
                    disable=False,
                    leave=True,
                    bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
                )
            else:
                val_loop = self.val_loader

            for step, batch in enumerate(val_loop):
                loss, losses, metrics = self._process_batch(step, batch)

                self.metrics.update_loss(loss, **losses)
                self.metrics.update_metrics(**metrics)
                avg_loss, mean_losses, mean_metrics = self.metrics.get_metrics()

                metrics = {"loss": avg_loss}
                metrics.update(mean_losses)
                metrics.update(mean_metrics)

                if epoch % 5 == 0:
                    val_loop.set_postfix(metrics)

        return self.metrics.loss / len(self.val_loader), metrics

    def run(self):
        score = self.__best_score
        for epoch in range(self.__epochs):
            print(f"Epoch {epoch + 1}/{self.__epochs}")

            if self.run_train:
                train_loss = self._train_loop(epoch)

            if self.run_val:
                val_loss, metrics = self._val_loop(epoch)
                if self.scheduler is not None:
                    for _, sdlr in self.scheduler.items():
                        sdlr.step(val_loss)  # Or use metrics?

            if (score > self.__best_score or not self.run_val) and self.run_train:
                self.__best_score = score
                self.__early_stop = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_path, f"{epoch:06}.pth"))

            elif self.run_train:
                print(f"Train loss: {train_loss:.4f}")
                self.__early_stop += 1

            if self.__early_stop >= self.__patience:
                print("Early stopping triggered. Stopping training.")
                break


class StyleGAN2Trainer(Trainer):
    def __init__(
        self, model: StyleGAN2, epochs: int, lr: float, train_loader: DataLoader, run_train: bool, device: torch.device
    ):
        super(StyleGAN2Trainer, self).__init__(
            model=model,
            epochs=epochs,
            lr=lr,
            train_loader=train_loader,
            val_loader=None,
            device=device,
            run_train=run_train,
            run_val=False,
            optimizer={
                "generator": optim.Adam(self.model.mapping_network.parameters() + self.model.generator.parameters()),
                "discriminator": optim.Adam(self.model.discriminator.parameters()),
            },
            scheduler=None,
            patience=100,
            output_path=None,
        )

        # Training Parameters
        self.gradient_penalty_coefficient = 10
        self.lazy_gradient_penalty_interval = 4
        self.lazy_path_penalty_interval = 32
        self.pl_weight = 2.0
        self.train_step_counter = 0
        self.pl_mean_var = torch.tensor(0.0)

        # params.log2res = 9
        # params.BATCH_SIZE = 32

    def gradient_penalty(self, reals: torch.Tensor) -> torch.Tensor:

        # Pass real samples to discriminator
        reals.requires_grad = True
        real_scores_out: torch.Tensor = self.model.discriminator(reals)
        # Calculate gradients with respect to real samples
        real_grads, *_ = torch.autograd.grad(
            outputs=real_scores_out.sum(), inputs=reals, create_graph=True, retain_graph=True
        )
        # Compute gradient penalty
        # https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/training/loss.py#L138C9-L138C20
        norms = real_grads.square().sum(dim=(1, 2, 3)).sqrt()
        gradient_penalty = (norms - 1.0).square()
        return gradient_penalty * (self.gradient_penalty_coefficient / (1.0**2))

    def path_length_penalty(
        self,
        const_input: torch.Tensor,
        fake_images_out: torch.Tensor | None = None,
        w_mapping: torch.Tensor | None = None,
    ):

        B = const_input.shape[0]
        L = self.model.latent_dim
        assert B == self.model.batch_size
        # Generate fake images
        if fake_images_out is None:
            B = min(2, B // 4)  # Reduce memory footprint
            w_mapping: torch.Tensor = self.model.mapping_network(z=torch.randn(B, L, device=const_input.device))
            w_mapping.requires_grad = True
            noise = [
                torch.randn((2, B, 2**res, 2**res, 1), device=const_input.device)
                for res in range(2, self.model.log2_end_res + 1)
            ]
            fake_images_out: torch.Tensor = self.model.generator(
                x=const_input, w=w_mapping, noise=noise
            )  # -> [B, C, H, W]
        assert w_mapping is not None, "Pass w_mapping aswell!"

        # Compute |J*y|
        pl_noise = torch.randn_like(fake_images_out, device=fake_images_out.device) / torch.sqrt(
            fake_images_out[0, 0, :, :].numel()
        )
        gradients, *_ = torch.autograd.grad(
            outputs=(fake_images_out * pl_noise).sum(),
            inputs=w_mapping,
            create_graph=True,
            grad_outputs=torch.ones(fake_images_out.shape, device=fake_images_out.device),
        )
        # https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/training/loss.py#L169
        pl_lengths = gradients.square().sum(dim=2).mean(dim=1).sqrt()  # calculate the l2-norm
        # Track exponential moving average of |J*y|.
        pl_mean = self.pl_mean_var + 0.01 * (pl_lengths.mean() - self.pl_mean_var)
        self.pl_mean_var = pl_mean.detach().clone()
        # Calculate (|J*y|-a)^2
        pl_penalty = (pl_lengths - pl_mean).square()
        # Weight regularization
        return pl_penalty * self.pl_weight

    def _process_batch(
        self, step: int, batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:

        if self.model.training:
            # Pass batch to model
            output_dict: dict[str, torch.Tensor] = self.model(batch)
            # Apply gradient penalty to discriminator_loss
            apply_gradient_penalty = self.train_step_counter % self.lazy_gradient_penalty_interval == 0
            # Apply path length penalty to generator_loss
            path_length_penalty = (
                self.train_step_counter >= 5000 & self.train_step_counter % self.lazy_path_penalty_interval == 0
            )
            # Retrive tensors
            discriminate_real = output_dict["discriminate_real"]
            discriminate_fake = output_dict["discriminate_fake"]
            const_input = output_dict["const_input"]
            # Generator Loss
            generator_loss = F.softplus(-discriminate_fake)  # .mean()
            if path_length_penalty:
                generator_loss += self.path_length_penalty(const_input)
            # Discriminator Loss
            # real_loss = F.softplus(-discriminate_real)
            # fake_loss = F.softplus(discriminate_fake)
            # discriminator_loss = real_loss + fake_loss  # torch.mean(real_loss) + torch.mean(fake_loss)
            discriminator_loss = discriminate_fake - discriminate_real
            discriminator_loss += discriminate_real.square() * 0.001
            if apply_gradient_penalty:
                discriminator_loss += self.gradient_penalty(batch)
            # Get metrics and losses
            real_accuracy = (hard_step(discriminate_real) == 1.0).float().mean().item()
            fake_accuracy = (hard_step(discriminate_fake) == 0.0).float().mean().item()
            self.train_step_counter += 1
            self.model.ada.update(discriminate_real)
        else:
            raise NotImplementedError("Validation not implemented!")

        return (
            generator_loss + discriminator_loss,
            {"generator_loss": generator_loss, "discriminator_loss": discriminator_loss},
            {"real_accuracy": real_accuracy, "fake_accuracy": fake_accuracy, "ada_p": self.model.ada.probability},
        )


if __name__ == "__main__":
    pass
