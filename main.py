import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms

from time import time

from models import *

# Define transformation to convert images to tensors with float32 dtype
def to_tensor():
    return transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float, scale=True),
        ])

def main():
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=to_tensor(),
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=to_tensor(),
    )

    batch_size = 128
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = Model(device=device)


    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    epochs = 1

    start_time = time()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
    print(f"Done! Training time: {time() - start_time:.2f} seconds")

def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X) # gets a matrix of predictions
        loss = loss_fn(pred, y) # computes the loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            accuracy = 100 * correct / len(X)
            print(f"loss: {loss:>7f}  accuracy: {accuracy:>5.1f}%  [{current:>5d}/{len(dataloader.dataset):>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    main()