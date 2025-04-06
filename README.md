# Simple Image Classifier with Python Neural Network

This project implements a basic Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset, based on the Medium article "A Simple Image Classifier with a Python Neural Network" by Data Bistrot.

## Overview

The project builds a CNN to classify 32x32 color images into 10 different classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Features

- Loads and preprocesses the CIFAR-10 dataset
- Visualizes sample training images
- Implements a CNN with three convolutional layers
- Trains the model and displays training progress
- Evaluates model performance on test data
- Makes predictions on sample test images

## Requirements

- Python 3.x
- TensorFlow
- Matplotlib
- NumPy

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install the required dependencies:
```bash
pip install tensorflow matplotlib numpy
```

## Usage

Run the main script:
```bash
python image_classifier.py
```

The script will:
1. Load and preprocess the CIFAR-10 dataset
2. Display sample images
3. Build and train the CNN model
4. Show training progress graphs
5. Display test accuracy
6. Show predictions for sample test images

## Model Architecture

The CNN consists of:
- 3 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- Flatten layer
- 2 Dense layers
- Softmax output layer

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)            (None, 30, 30, 32)        896       
 max_pooling2d (MaxPooling2D) (None, 15, 15, 32)       0         
 conv2d_1 (Conv2D)          (None, 13, 13, 64)        18496     
 max_pooling2d_1 (MaxPooling2D) (None, 6, 6, 64)        0         
 conv2d_2 (Conv2D)          (None, 4, 4, 64)          36928     
 flatten (Flatten)          (None, 1024)              0         
 dense (Dense)              (None, 64)                65600     
 dense_1 (Dense)            (None, 10)                650       
=================================================================
Total params: 122,570
Trainable params: 122,570
Non-trainable params: 0

## Results

- Training time: ~few minutes (depending on hardware)
- Expected test accuracy: ~60-70% after 10 epochs
- Visualizations of training progress and sample predictions

## Contributing

Feel free to fork this repository and submit pull requests with improvements or modifications.

## License

This project is licensed under the MIT License.

## Acknowledgments

Based on the Medium article by Data Bistrot: [A Simple Image Classifier with a Python Neural Network](https://medium.com/data-bistrot/a-simple-image-classifier-with-a-python-neural-network-82a5522fe48b)
