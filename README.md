# MNIST Digit Classifier (CNN - PyTorch)

A Convolutional Neural Network (CNN) built from scratch using PyTorch to classify handwritten digits from the MNIST dataset.

## Features
- Custom CNN architecture using Conv2D and MaxPooling
- Trained using CrossEntropyLoss and Adam optimizer
- GPU-accelerated training
- Achieves ~99% accuracy on test data

## Model Architecture
Input (1×28×28) →
Conv → ReLU → MaxPool →
Conv → ReLU → MaxPool →
Fully Connected → 10 classes

## Results
- Training Accuracy: ~99%
- Test Accuracy: ~98–99%

## How to Run
```bash
pip install -r requirements.txt
python train.py
