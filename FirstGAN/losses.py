import torch
import torch.nn as nn
from torch import Tensor
from utils import get_noise
from models import Generator, Discriminator

class GANLoss(nn.Module):
    def __init__(self, 
                generator: Generator, 
                discriminator: Discriminator,
                device) -> None:
        super().__init__()
        self.gen = generator
        self.disc = discriminator
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device

    def forward(self, num_images: int, noise_dim: int, real_images: Tensor = None, step: str = "disc"):
        if step == "disc":
            return self._get_discriminator_loss(real_images=real_images, num_images=num_images, noise_dim=noise_dim)
        else:
            return self._get_generator_loss(num_images=num_images, noise_dim=noise_dim)

    def _get_discriminator_loss(self, real_images, num_images, noise_dim):
        noise_vectors = get_noise(n_samples=num_images, noise_dim=noise_dim, device=self.device)
        fake_images = self.gen(noise_vectors)

        # Detach the generator because we don't want to include gen's parameters in training the discriminator.
        disc_fake_output = self.disc(fake_images.detach())
        fake_ground_truth = torch.zeros_like(disc_fake_output)
        fake_loss = self.criterion(disc_fake_output, fake_ground_truth)
        disc_real_output = self.disc(real_images)
        real_ground_truth = torch.ones_like(disc_real_output)
        real_loss = self.criterion(disc_real_output, real_ground_truth)
        disc_loss = (fake_loss + real_loss) / 2
        return disc_loss

    def _get_generator_loss(self, num_images, noise_dim):
        noise_vectors = get_noise(n_samples=num_images, noise_dim=noise_dim)
        fake_images = self.gen(noise_vectors)
        disc_output = self.disc(fake_images)
        ground_truth = torch.ones_like(disc_output)
        gen_loss = self.criterion(disc_output, ground_truth)
        return gen_loss