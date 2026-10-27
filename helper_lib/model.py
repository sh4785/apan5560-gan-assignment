"""GAN architecture for 28x28 MNIST digits (Assignment 3, Part 1)."""
import torch
import torch.nn as nn

LATENT_DIM = 100  # size of the noise vector z


class Generator(nn.Module):
    """z (B, 100) -> image (B, 1, 28, 28) in [-1, 1]."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        # Fully connected layer: 100 -> 7*7*128
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.deconv = nn.Sequential(
            # (B,128,7,7) -> (B,64,14,14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # (B,64,14,14) -> (B,1,28,28)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)  # reshape to (B, 128, 7, 7)
        return self.deconv(x)


class Discriminator(nn.Module):
    """image (B, 1, 28, 28) -> probability image is real (B, 1)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (B,1,28,28) -> (B,64,14,14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (B,64,14,14) -> (B,128,7,7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Flatten -> Linear -> single real/fake probability
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def weights_init(m: nn.Module) -> None:
    """DCGAN-style init: N(0, 0.02) for conv/linear, N(1, 0.02) for BatchNorm."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def get_model(model_name: str):
    """Factory used across the helper library. 'GAN' returns (generator, discriminator)."""
    name = model_name.lower()
    if name == "gan":
        generator, discriminator = Generator(), Discriminator()
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        return generator, discriminator
    raise ValueError(f"Unknown model name: {model_name}")
