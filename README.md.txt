# CIFAR-10 Image Classification with CNN (PyTorch)

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset. The model includes data augmentation, dropout, early stopping, and learning rate scheduling to improve performance and prevent overfitting.

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Requirements](#requirements)

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

We use `torchvision.datasets.CIFAR10` to download and preprocess the dataset.

## Model Architecture
The CNN model has:
- 3 convolutional blocks with Conv2d, BatchNorm, ReLU, MaxPool2d, and Dropout2d
- Fully connected layers with ReLU and Dropout
- Dropout used to reduce overfitting
- CrossEntropyLoss as the criterion

```python
Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout
... (repeated for deeper layers)
Flatten -> Linear -> ReLU -> Dropout -> Linear -> Output

##Training
Optimizer: Adam (lr=0.001)
Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
Early stopping with patience=5
Data augmentation: RandomHorizontalFlip, RandomCrop, ColorJitter
Train-validation split: 80%-20%
Batch size: 64

##Evaluation

The model is evaluated on the test set for:

Accuracy
Confusion matrix
Precision, recall, and F1-score per class