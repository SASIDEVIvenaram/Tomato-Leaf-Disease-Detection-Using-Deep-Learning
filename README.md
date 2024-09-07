# Tomato Leaf Disease Detection Using Deep Learning

This project utilizes a Convolutional Neural Network (CNN) model built with TensorFlow and Keras to detect and classify various tomato leaf diseases. The model is trained on the PlantVillage dataset, which consists of images of healthy and diseased tomato leaves. This repository contains code for data preprocessing, model training, evaluation, and testing on new images.

## Features

- Image classification: The model is capable of predicting whether a tomato leaf is healthy or affected by a disease.

- Model architecture: CNN architecture with multiple convolutional and pooling layers.

- Data augmentation: Techniques such as random flipping and rotation to improve model generalization.

- Model evaluation: Visualized results for training and validation accuracy and loss.

- Image prediction demo: A utility to predict disease from a new image.

## Dataset
The model is trained on the publicly available [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) dataset, which contains labeled images of various plant diseases. Specifically, we are using the tomato leaf subset for this project.

## Classes
The dataset includes various diseases, such as:

- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Mosaic Virus
- Tomato Septoria Leaf Spot
- Tomato Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Healthy

# Getting Started

## Prerequisites

- Python 3.7 or later
- TensorFlow 2.x
- Keras
- Matplotlib
- Pandas & NumPy

## Model Architecture
The CNN model includes:

- Convolutional layers followed by max pooling layers
- Data augmentation layers such as random flipping and rotation
- Dense layers with ReLU activation
- Softmax layer for multi-class classification

## Results
- Training Accuracy: ~95% (after 50 epochs)
- Validation Accuracy: ~92%
- Test Accuracy: ~90%

The following charts show the training/validation accuracy and loss during model training:


![Image](https://github.com/user-attachments/assets/a0cfb6a8-3403-4896-8077-0074735e86be)
