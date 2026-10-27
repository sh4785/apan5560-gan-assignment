"""Data loading for MNIST."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loader(data_dir: str = "./data", batch_size: int = 128, train: bool = True) -> DataLoader:
    """Return an MNIST DataLoader with pixels scaled to [-1, 1] (matches the generator's Tanh)."""
    transform = transforms.Compose([
        transforms.ToTensor(),                  # [0, 255] -> [0, 1]
        transforms.Normalize((0.5,), (0.5,)),   # [0, 1]  -> [-1, 1]
    ])
    dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0, drop_last=train)
