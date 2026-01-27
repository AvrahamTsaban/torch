# PyTorch MNIST & Fashion-MNIST Classifier

A PyTorch-based image classification project for training neural networks on MNIST and Fashion-MNIST datasets, with support for custom datasets.

## Project Overview

This project implements several neural network architectures to classify handwritten digits (MNIST) and fashion items (Fashion-MNIST). It includes functionality to train models, evaluate them on standard test sets, and test on custom user-provided images.

## Project Structure

```
.
├── main.py              # Main training and evaluation pipeline
├── models.py            # Neural network architecture definitions
├── datasets.py          # Dataset loading and preprocessing
├── custom_dataset.py    # Custom dataset class and image preprocessing
├── data/
│   ├── MNIST/          # MNIST dataset (auto-downloaded)
│   ├── FashionMNIST/   # Fashion-MNIST dataset (auto-downloaded)
│   ├── myNums/         # Custom handwritten digit images
│   └── myFashion/      # Custom fashion item images
└── examples/           # Example outputs and results
```

## Features

- **Multiple Model Architectures**: Three progressively larger neural networks
  - [`Base_Model`](models.py): Lightweight model (51,210 parameters)
  - [`Long_Model`](models.py): Medium model (3,156,490 parameters)
  - [`Giant_Model`](models.py): Large model (536,928,778 parameters)

- **Dataset Support**:
  - MNIST handwritten digits
  - Fashion-MNIST clothing items
  - Custom user-provided images via [`CustomDataset`](custom_dataset.py)

- **Image Preprocessing**: [`mnistify`](custom_dataset.py) function converts any image to MNIST format (28×28 grayscale, inverted, autocontrasted)

- **Training Features**:
  - Automatic device detection (CPU/CUDA)
  - AdamW optimizer with weight decay
  - Cross-entropy loss
  - Accuracy and loss tracking per epoch

- **Visualization**: [`plot_results`](main.py) function generates accuracy graphs over training epochs

## Quick Start

### Installation

```bash
pip install torch torchvision pillow matplotlib
```

### Running the Project

```python
python main.py
```

This will:
1. Train [`Long_Model`](models.py) on Fashion-MNIST for 10 epochs
2. Evaluate on built-in test set and custom images
3. Display accuracy plots
4. Repeat the process for MNIST digits

### Using Custom Images

**Requirements:**
1. Place images in `data/myNums/` or `data/myFashion/`
2. **Name files with the label as the first character** (e.g., `3_handwritten.png` for digit 3)
3. **Images must have a white or transparent background**
4. Supported format: PNG files
5. The [`mnistify`](custom_dataset.py) function will automatically preprocess them to MNIST format

## API Reference

### Main Functions

#### [`deep_learning()`](main.py)
Trains and tests a model on given datasets.

**Parameters:**
- `train_d`: Training dataset
- `test_d1`: Built-in test dataset
- `test_d2`: Custom test dataset (optional)
- `model`: Model class (default: [`Base_Model`](models.py))
- `device`: "cpu" or "cuda" (default: "cpu")
- `epochs`: Number of training epochs (default: 5)

**Returns:** `(train_results, test_results, test2_results)`

#### [`train()`](main.py)
Trains the model for one epoch.

**Returns:** Training accuracy for the epoch

#### [`test()`](main.py)
Evaluates model on test dataset.

**Parameters:**
- `verbose`: If `True`, saves classified images to `{dataset.path}/classified/{predicted_label}/sample_{i}.png`

**Returns:** Test accuracy

### Custom Dataset

#### [`CustomDataset`](custom_dataset.py)
PyTorch Dataset for loading and preprocessing custom images.

**Parameters:**
- `path`: Directory containing PNG images
- `transform`: Optional torchvision transform
- `img_process`: Preprocessing function (default: [`mnistify`](custom_dataset.py))

**Image Naming Convention:** First character of filename is the label (e.g., `7_shoe.png` → label 7)

#### [`mnistify()`](custom_dataset.py)
Converts any PIL image to MNIST format:
- Converts to 28×28 grayscale
- Inverts colors (white background → black background)
- Applies autocontrast
- Crops to bounding box
- Resamples using Lanczos interpolation

### Models

All models inherit from [`Base_Model`](models.py) and use a flatten layer followed by fully connected layers with ReLU activation.

- **[`Base_Model`](models.py)**: 28×28 → 64 → 32 → 10
- **[`Long_Model`](models.py)**: 28×28 → 1024 → 1024 → 1024 → 10
- **[`Giant_Model`](models.py)**: 28×28 → 16384 → 16384 → 8192 → 512 → 10

## Example Usage

### Training a Custom Model

```python
from models import Base_Model
from datasets import number_train, number_test
from main import deep_learning

train_acc, test_acc, _ = deep_learning(
    train_d=number_train,
    test_d1=number_test,
    model=Base_Model,
    device="cuda",
    epochs=10
)
```

### Processing Custom Images

```python
from custom_dataset import mnistify
from PIL import Image

with Image.open("my_digit.png") as img:
    processed = mnistify(img)
    processed.save("my_digit_mnist.png")
```

### Using the Verbose Test Mode

```python
from main import test
from datasets import my_number_test
from torch.utils.data import DataLoader

# This will save classified images to data/myNums/classified/{label}/
test(
    DataLoader(my_number_test, batch_size=32),
    model,
    loss_fn,
    device="cpu",
    verbose=True
)
```

## Available Datasets

From [`datasets.py`](datasets.py):
- [`fashion_train`](datasets.py) / [`fashion_test`](datasets.py): FashionMNIST
- [`number_train`](datasets.py) / [`number_test`](datasets.py): MNIST
- [`my_fashion_test`](datasets.py): Custom fashion images
- [`my_number_test`](datasets.py): Custom digit images

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE)

## Notes

- Images are automatically downloaded on first run
- Training uses AdamW optimizer (lr=1e-3, weight_decay=1e-3)
- Batch size is set to 128
- Models automatically move to available accelerator (CUDA if available)
- Special thanks to claude sonnet for documenting this project in a minute