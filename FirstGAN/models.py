import torch.nn as nn
from torch import Tensor

class Generator(nn.Module):
    def __init__(self, noise_dim: int = 10, image_dim: int = 7, hidden_dim: int = 128) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self._make_generator()

    def _create_generator_block(self, input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        
    def _make_generator(self) -> None:
        self.gen = nn.Sequential(
            self._create_generator_block(input_dim=self.noise_dim, output_dim=self.hidden_dim),
            self._create_generator_block(self.hidden_dim, self.hidden_dim * 2),
            self._create_generator_block(self.hidden_dim * 2, self.hidden_dim * 4),
            self._create_generator_block(self.hidden_dim * 4, self.hidden_dim * 8),
            nn.Linear(self.hidden_dim * 8, self.image_dim),
            nn.Sigmoid()
        )

    def forward(self, noise) -> Tensor:
        return self.gen(noise)

    def _get_generator(self) -> nn.Sequential:
        return self.gen


class Discriminator(nn.Module):
    def __init__(self, image_dim: int = 784, hidden_dim: int = 128) -> None:
        super().__init__()
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self._make_discriminator()
        
    def _create_discriminator_block(self, input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2)
        )

    def _make_discriminator(self) -> None:
        self.disc = nn.Sequential(
            self._create_discriminator_block(self.image_dim, self.hidden_dim * 4),
            self._create_discriminator_block(self.hidden_dim * 4, self.hidden_dim * 2),
            self._make_discriminator(self.hidden_dim * 2, self.hidden_dim),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x) -> Tensor:
        return self.disc(x)

    def _get_discriminator(self):
        return self.disc