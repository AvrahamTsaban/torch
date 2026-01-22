from tkinter.filedialog import test
import torch
from torch import nn
from torch.utils.data import DataLoader
from time import time

from models import *
from datasets import *

def deep_learning(train_d, test_d1, test_d2=None, model=Base_Model, device="cpu", epochs=5):

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
            a = test(test2_dataloader, model, loss_fn, device, name="Custom Test")
            test2_results.append(a)
        print()
        
    print(f"Done! Training time: {time() - start_time:.2f} seconds")
    return train_results, test_results, test2_results if test_d2 else None

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
    return accuracy

def test(dataloader, model, loss_fn, device, name="Test", verbose=False):
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
                print(f"Predicted: {pred[i].argmax().item()}, actual: {y[i].item()}")

    loss = test_loss / len(dataloader)
    accuracy = 100 * correct / len(dataloader.dataset)
    print(f"{name}: accuracy - {accuracy:>5.1f}%, average loss - {loss:>8f}")
    return accuracy

    # תרשים דיוק לאורך epochs
def plot_results(train, data, real_data=None, test_name="Test", model_name="Model"):
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
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    train, data, real_data = deep_learning(train_d=fashion_train, test_d1=fashion_test, test_d2=my_fashion_test, device=device, model=Giant_Model, epochs=10)
    plot_results(train, data, real_data, test_name="Fashion Test", model_name="Giant_Model")

    train, data, real_data = deep_learning(train_d=number_train, test_d1=number_test, test_d2=my_number_test, device=device, model=Giant_Model, epochs=10)
    plot_results(train, data, real_data, test_name="Number Test", model_name="Giant_Model")

if __name__ == "__main__":
    main()