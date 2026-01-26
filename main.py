import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from tkinter.filedialog import test
from time import time
import os

from models import *
from datasets import *

def deep_learning(train_d, test_d1, test_d2=None, model=Base_Model, device="cpu", epochs=5):
    """Trains and tests a model on the given datasets.
    Args:
        train_d: training dataset
        test_d1: built-in test dataset
        test_d2: custom test dataset (optional)
        model: model class to instantiate
        device: device to use ("cpu" or "cuda")
        epochs: number of training epochs
    Returns:
        train_results: list of training accuracies per epoch
        test_results: list of built-in test accuracies per epoch
        test2_results: list of custom test accuracies per epoch (if test_d2 is provided)
    """

    batch_size = 128
    train_dataloader = DataLoader(train_d, batch_size=batch_size)
    test_dataloader = DataLoader(test_d1, batch_size=batch_size)
    if test_d2:
        test2_dataloader = DataLoader(test_d2, batch_size=batch_size)
    
    
    model = model(device=device)

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    start_time = time()
    train_results = []
    test_results = []
    if test_d2:
        test2_results = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        a = train(train_dataloader, model, loss_fn, optimizer, device)
        train_results.append(a)
        a = test(test_dataloader, model, loss_fn, device, name="Built in Test")
        test_results.append(a)
        if test_d2:
            a = test(test2_dataloader, model, loss_fn, device, name="Custom Test", verbose=True)
            test2_results.append(a)
        print()
        
    print(f"Done! Training time: {time() - start_time:.2f} seconds")
    return train_results, test_results, test2_results if test_d2 else None

def train(dataloader, model, loss_fn, optimizer, device):
    """Trains the model for one epoch on the given dataloader.
    Args:
        dataloader: DataLoader for the training dataset
        model: model to train
        loss_fn: loss function
        optimizer: optimizer for updating model parameters
        device: device to use ("cpu" or "cuda")
    Returns:
        accuracy: training accuracy for the epoch
        (under the hood, also updates the model parameters)
    """
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
    return accuracy

def test(dataloader, model, loss_fn, device, name="Test", verbose=False):
    """Tests the model on the given dataloader.
    Args:
        dataloader: DataLoader for the test dataset
        model: model to test
        loss_fn: loss function
        device: device to use ("cpu" or "cuda")
        name: name of the test (for printing purposes)
        verbose: whether to print individual predictions and save images, classified by folders ("predicted_label/sample_i.png"). 
        - Don't use for datasets larger than a few samples. 
        - Default is False.
    Returns:
        accuracy: test accuracy
    """
    model.eval()
    torch.set_grad_enabled(False)
    test_loss, correct = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        test_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if verbose:
            for i in range(len(y)):
                predict = pred[i].argmax().item()
                print(f"Predicted: {predict}, actual: {y[i].item()}")
                outpath = os.path.join(dataloader.dataset.path, "classified", str(predict), f"sample_{i}.png")
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                save_image(X[i], outpath)
                

    loss = test_loss / len(dataloader)
    accuracy = 100 * correct / len(dataloader.dataset)
    print(f"{name}: accuracy - {accuracy:>5.1f}%, average loss - {loss:>8f}")
    return accuracy

    # תרשים דיוק לאורך epochs
def plot_results(train, data, real_data=None, test_name="Test", model_name="Model"):
    """Plots the accuracy results over epochs.
    Args:
        train: list of training accuracies per epoch
        data: list of built-in test accuracies per epoch
        real_data: list of custom test accuracies per epoch (optional)
        test_name: name of the test (for title)
        model_name: name of the model (for title)
    Disclosure for teacher:
        this function was completely written by copilot; I should get no credit for it.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train) + 1), train, label='Train', marker='o')
    plt.plot(range(1, len(data) + 1), data, label='Built-in Test', marker='s')
    if real_data:
        plt.plot(range(1, len(real_data) + 1), real_data, label='Custom Test', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{test_name}: Accuracy over Epochs for {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """Main function to run training and testing on different datasets and models.
    - Trains and tests Giant_Model (example) on Fashion-MNIST dataset.
    - Plots results.
    - Repeats the process for the Number dataset."""
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    train, data, real_data = deep_learning(train_d=fashion_train, test_d1=fashion_test, test_d2=my_fashion_test, device=device, model=Long_Model, epochs=10)
    plot_results(train, data, real_data, test_name="Fashion Test", model_name="Long_Model")

    train, data, real_data = deep_learning(train_d=number_train, test_d1=number_test, test_d2=my_number_test, device=device, model=Long_Model, epochs=10)
    plot_results(train, data, real_data, test_name="Number Test", model_name="Long_Model")

if __name__ == "__main__":
    main()