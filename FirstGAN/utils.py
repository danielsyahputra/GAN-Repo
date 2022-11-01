import torch
from torch import Tensor
from typing import Tuple
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

torch.manual_seed(0)

def show_tensor_images(image_tensor: Tensor, num_images: int = 25, size: Tuple = (1, 28, 28)) -> None:
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def get_noise(n_samples: int, noise_dim: int, device:str = "cpu") -> Tensor:
    return torch.randn(n_samples, noise_dim).to(device)