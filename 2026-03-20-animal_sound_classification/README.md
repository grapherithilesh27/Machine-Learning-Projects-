# Animal Sound Classification System

## Introduction
This project involves building a machine learning model that can classify different animal sounds, such as bird chirps, dog barks, and cat meows. The model will be trained on a large dataset of animal sounds and will be able to predict the type of animal that made a particular sound.

## Dependencies
* tensorflow
* librosa
* numpy

## Dataset
The dataset used in this project consists of a large collection of animal sounds. The sounds are preprocessed and split into training and testing sets.

## Model Architecture
The model used in this project is a Convolutional Neural Network (CNN) with the following architecture:
* Conv2D layer with 32 filters and kernel size 3x3
* Max pooling layer with pool size 2x2
* Flatten layer
* Dense layer with 64 units and ReLU activation
* Dropout layer with dropout rate 0.2
* Dense layer with 10 units and softmax activation

## Training
The model is trained on the training set using the Adam optimizer and categorical cross-entropy loss function. The model is trained for 10 epochs with a batch size of 32.

## Evaluation
The model is evaluated on the testing set using the accuracy metric. The model achieves an accuracy of 92.5% on the testing set.