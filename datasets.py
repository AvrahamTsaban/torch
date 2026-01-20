from torchvision import datasets
import torchvision.transforms.v2 as transforms
from custom_dataset import CustomDataset
import torch

# Define transformation to convert images to tensors with float32 dtype
def to_tensor():
    return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float, scale=True),
        ])

"""
datasets for training and testing:
- fashion_train: FashionMNIST training dataset
- fashion_test: FashionMNIST test dataset
- number_train: MNIST training dataset
- number_test: MNIST test dataset
- my_fashion_test: CustomDataset for user-provided fashion images, processed to MNIST format
- my_number_test: CustomDataset for user-provided number images, processed to MNIST format
"""
fashion_train = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=to_tensor(),
    )

fashion_test = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=to_tensor(),
    )

number_train = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=to_tensor(),
    )

number_test = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=to_tensor(),
    )

my_fashion_test = CustomDataset(
        path="./data/myFashion",
        transform=to_tensor(),
    )

my_number_test = CustomDataset(
        path="./data/myNums",
        transform=to_tensor(),
    )