import torch
from torch import nn
from torch.utils.data import DataLoader
from time import time

from models import *
from datasets import *

def deep_learning(train_d, test_d1, test_d2=None, model=Base_Model, device="cpu"):

    batch_size = 128
    train_dataloader = DataLoader(train_d, batch_size=batch_size)
    test_dataloader = DataLoader(test_d1, batch_size=batch_size)
    if test_d2:
        test2_dataloader = DataLoader(test_d2, batch_size=batch_size)
    
    
    model = model(device=device)

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    epochs = 1

    start_time = time()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        if test_d2:
            test(test_dataloader, model, loss_fn, device, name="Test1")
            test(test2_dataloader, model, loss_fn, device, name="Test2")
        else:
            test(test_dataloader, model, loss_fn, device)
        print()
    print(f"Done! Training time: {time() - start_time:.2f} seconds")

def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    torch.set_grad_enabled(True)
    train_loss, correct = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X) # gets a matrix of predictions
        loss = loss_fn(pred, y) # computes the loss

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    loss = train_loss / len(dataloader)
    accuracy = 100 * correct / len(dataloader.dataset)
    print(f"Train: accuracy - {accuracy:>5.1f}%, average loss - {loss:>7f}")

def test(dataloader, model, loss_fn, device, name="Test"):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss, correct = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        test_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss = test_loss / len(dataloader)
    accuracy = 100 * correct / len(dataloader.dataset)
    print(f"{name}: accuracy - {accuracy:>5.1f}%, average loss - {loss:>8f}")


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    deep_learning(train_d=number_train, test_d1=number_test, test_d2=my_fashion_test, device=device, model=Long_Model)