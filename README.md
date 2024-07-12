
# Convolutional-Neural-Network-Image-Processor

This a an end to end capstone project on CNN image classifier. The source of the data is [Kaggle](#https://www.kaggle.com/datasets/alvarogarciav/dataset-classifier-cat-dog-snake) website. The project has been divided into following parts:
1. Data preporcessing
2. Model training
3. Model Deployment

## Cat, Dog, and Snake Image Classifier with Streamlit Deployment
### Project Overview

This repository contains a CNN-based image classifier trained to distinguish between cat, dog, and snake images. The model is deployed using Streamlit for a user-friendly interface. 

## Features

* CNN Model: A Convolutional Neural Network trained on a dataset of cat, dog, and snake images.  
* Streamlit App: A web-based application to upload and classify images in real-time.  
* Clear and Concise Code: Well-structured Python code for model training and deployment.  

## CNN Model Description
### Architecture:

The described CNN model consists of the following layers:

### Convolutional Layers
* Three convolutional layers, each with 32 kernels of size 3x3.
* These layers extract local features from the input image.
* ReLU activation function is applied to introduce non-linearity.
### Max Pooling Layer
* A max pooling layer with a pool size of 2x2 and a stride of 2.
* Downsamples the feature maps, reducing computational complexity and providing some invariance to small translations and distortions.
### Fully Connected Layers
* Two fully connected (dense) layers.
* The first layer has 128 nodes.
* The second (output) layer has 3 nodes, corresponding to the number of classes (cat, dog, snake).
* ReLU activation is used in the first fully connected layer, while softmax activation is applied to the output layer for probability distribution generation.
### Overall Function:

The model processes an input image through the convolutional layers to extract relevant features. The max pooling layer reduces dimensionality while preserving important information. The extracted features are then flattened and fed into the fully connected layers for classification. The final softmax output provides probabilities for each of the three classes.



**Click on the Youtube link given below**
[![Watch the video](https://img.youtube.com/vi/oWmAqrceugM/maxresdefault.jpg)](https://www.youtube.com/watch?v=oWmAqrceugM)
