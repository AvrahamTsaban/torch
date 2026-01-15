import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Define transformation to convert images to tensors with float16 dtype
def to_tensor():
    return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float16, scale=True),
        ])

def main():
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=to_tensor(),
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=to_tensor(),
    )

    batch_size = 16
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = Model().to(device)
    print(f"Using {device} device")
    print(model)

if __name__ == "__main__":
    main()