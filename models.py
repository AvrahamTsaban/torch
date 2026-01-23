from torch import nn
from torchvision.models import vgg16 as vgg16

"""Contains different model architectures for training."""

class Base_Model(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        self.to(device)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Long_Model(Base_Model):
    def __init__(self, device="cpu"):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )
        self.to(device)

class Giant_Model(Base_Model):
    def __init__(self, device="cpu"):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 16384),
            nn.ReLU(),
            nn.Linear(16384, 16384),
            nn.ReLU(),
            nn.Linear(16384, 8192),
            nn.ReLU(),
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        self.to(device)

if __name__ == "__main__":
    models = [Base_Model, Long_Model, Giant_Model]
    for model in models:
        net = model()
        print(f"total parameters in {model.__name__}: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")