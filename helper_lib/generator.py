"""Sampling helpers: generate digits from a trained generator."""
import io
import math
from typing import Optional

import torch
from PIL import Image
from torchvision.utils import make_grid


@torch.no_grad()
def generate_samples(generator, num_samples: int = 16, device: str = "cpu",
                     latent_dim: int = 100, seed: Optional[int] = None) -> torch.Tensor:
    """Return a (num_samples, 1, 28, 28) tensor of generated digits in [0, 1]."""
    if seed is not None:
        torch.manual_seed(seed)
    generator.eval()
    z = torch.randn(num_samples, latent_dim, device=device)
    imgs = generator(z)
    imgs = (imgs + 1) / 2  # Tanh output [-1, 1] -> [0, 1]
    return imgs.clamp(0, 1).cpu()


def samples_to_png(imgs: torch.Tensor) -> bytes:
    """Arrange images in a square-ish grid and return PNG bytes."""
    nrow = math.ceil(math.sqrt(imgs.size(0)))
    grid = make_grid(imgs, nrow=nrow, padding=2)            # (3, H, W) in [0, 1]
    arr = grid.mul(255).byte().permute(1, 2, 0).numpy()     # (H, W, 3) uint8
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def save_samples_grid(imgs: torch.Tensor, path: str) -> None:
    with open(path, "wb") as f:
        f.write(samples_to_png(imgs))
