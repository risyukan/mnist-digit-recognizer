# MNIST Digit Recognizer

A simple and clear implementation of handwritten digit recognition using the MNIST dataset.
This project demonstrates a basic machine learning pipeline including data preprocessing, model training, evaluation, and visualization.

<img width="1800" height="663" alt="image" src="https://github.com/user-attachments/assets/ad22b5dc-3b3e-440c-a760-83d91800db5f" />


___

## 🧠 Overview

This project aims to classify handwritten digits (0–9) from the popular MNIST dataset. The implementation includes:

- Data loading and preprocessing
- A simple FNN network model (extended to a Convolutional Neural Network)
- Training and evaluation routines
- Visualization of predictions and loss curves
- Saving trained parameters

___

## ✨ Features
- Entire neural network implemented from scratch using only NumPy (no TensorFlow/PyTorch).
- Modular structure: layers, activation functions, optimizers, and training utilities are defined as reusable Python classes.
- Clear separation between dataset, model, and trainer for better maintainability.

___
## 📊 Dataset
- MNIST: 70,000 grayscale images of handwritten digits (28×28 pixels)
- Split into:
  - 60,000 training images
  - 10,000 test images
- Each image is labeled as an integer from 0 to 9.


## 🛠️Installation
This project requires Python 3.x and the following libraries:

```pip install numpy matplotlib```

(Optional: If you want to re-download the dataset directly)

```pip install torchvision```

___

## 🚀 Usage
Clone the repository and run the training script:
``` 
git clone https://github.com/risyukan/mnist-digit-recognizer.git
cd mnist-digit-recognizer
python mnist_train.py 
```

This will:
- Train the CNN on MNIST
- Save trained parameters to params.pkl
- Plot training and test accuracy curves
  
## 📈 Results
- Training accuracy: ~99%
- Test accuracy: ~98%

## Future Plans
- Build Diffusion and other deep learning models from scratch.


