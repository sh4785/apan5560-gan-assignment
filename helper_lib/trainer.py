"""Training loop for the GAN."""
import torch
import torch.nn as nn


def train_gan(generator, discriminator, data_loader, device="cpu", epochs=15, lr=2e-4, latent_dim=100):
    """Alternate one discriminator step and one generator step per batch.

    D maximizes log D(x) + log(1 - D(G(z)))
    G uses the non-saturating loss: maximize log D(G(z)), i.e. BCE(D(G(z)), 1)
    """
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history = {"d_loss": [], "g_loss": []}

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        d_running, g_running = 0.0, 0.0

        for real, _ in data_loader:
            real = real.to(device)
            b = real.size(0)
            real_labels = torch.ones(b, 1, device=device)
            fake_labels = torch.zeros(b, 1, device=device)

            # ---- 1) Train Discriminator ----
            z = torch.randn(b, latent_dim, device=device)
            fake = generator(z)
            d_loss_real = criterion(discriminator(real), real_labels)
            d_loss_fake = criterion(discriminator(fake.detach()), fake_labels)  # detach: don't update G here
            d_loss = d_loss_real + d_loss_fake

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # ---- 2) Train Generator ----
            g_loss = criterion(discriminator(fake), real_labels)  # G wants D to say "real"

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            d_running += d_loss.item()
            g_running += g_loss.item()

        n = len(data_loader)
        history["d_loss"].append(d_running / n)
        history["g_loss"].append(g_running / n)
        print(f"Epoch [{epoch}/{epochs}]  D loss: {d_running / n:.4f}  G loss: {g_running / n:.4f}")

    return history
