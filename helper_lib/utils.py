"""Save / load model weights."""
import os

import torch


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path: str, device: str = "cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
