import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from losses import GANLoss
from tqdm.auto import tqdm
from models import Generator, Discriminator
from utils import show_tensor_images, get_noise

def main():

    # Setup training

    epochs = 200
    noise_dim = 64
    display_step = 1000
    batch_size = 128
    lr = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(
        MNIST(".", download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True
    )
    
    # Setup model and optimizer
    generator = Generator(noise_dim=noise_dim).to(device)
    generator_opt = optim.Adam(generator.parameters(), lr=lr)
    discriminator = Discriminator().to(device)
    discriminator_opt = optim.Adam(discriminator.parameters(), lr=lr)

    current_step = 1
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    gan_loss = GANLoss(generator=generator, discriminator=discriminator, device=device)

    # Training loop
    for epoch in tqdm(range(1, epochs + 1)):
        for real, _ in loader:
            cur_batch_size = len(real)

            # Train discriminator
            real = real.view(cur_batch_size, -1).to(device)
            discriminator_opt.zero_grad()
            disc_loss = gan_loss.forward(num_images=cur_batch_size, noise_dim=noise_dim, real_images=real)
            disc_loss.backward(retain_graph=True)
            discriminator_opt.step()

            # Train generator
            generator_opt.zero_grad()
            gen_loss = gan_loss.forward(num_images=cur_batch_size, noise_dim=noise_dim, step="gen")
            gen_loss.backward()
            generator_opt.step()

            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            if current_step % display_step == 0 and current_step > 0:
                print(f"Step {current_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                noise_vector = get_noise(n_samples=cur_batch_size, noise_dim=noise_dim, device=device)
                fake_image = generator(noise_vector)
                show_tensor_images(fake_image)
                show_tensor_images(real)
                mean_discriminator_loss = 0
                mean_generator_loss = 0

if __name__=="__main__":
    main()