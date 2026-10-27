"""Train the GAN on MNIST and save the generator weights (Assignment 3, Part 2).

Usage:
    python train_gan.py              # 15 epochs, auto-picks GPU/Apple MPS/CPU
    python train_gan.py --epochs 5   # quicker test run
    python train_gan.py --device cpu # force CPU if MPS gives errors
"""
import argparse

import torch

from helper_lib import (get_mnist_loader, get_model, train_gan,
                        generate_samples, save_samples_grid, save_model)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon Macs
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = args.device or pick_device()
    print(f"Training on: {device}")

    loader = get_mnist_loader(batch_size=args.batch_size, train=True)
    generator, discriminator = get_model("GAN")

    train_gan(generator, discriminator, loader, device=device, epochs=args.epochs, lr=args.lr)

    # Save on CPU so the weights load anywhere (Docker, instructor's machine)
    generator.cpu()
    discriminator.cpu()
    save_model(generator, "models/gan_generator.pth")
    save_model(discriminator, "models/gan_discriminator.pth")
    print("Saved models/gan_generator.pth and models/gan_discriminator.pth")

    imgs = generate_samples(generator, num_samples=16, device="cpu", seed=0)
    save_samples_grid(imgs, "outputs/sample_grid.png")
    print("Saved sample digits to outputs/sample_grid.png")


if __name__ == "__main__":
    main()
