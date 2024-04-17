# Image Classification on CIFAR-10

This repository contains a Jupyter notebook that demonstrates the process of building a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset. CIFAR-10 is a dataset that consists of 60,000 32x32 color images in 10 different classes.

## Features

- Data visualization of CIFAR-10 images.
- Preprocessing of image data.
- Construction of a CNN model using TensorFlow and Keras.
- Implementation of custom callbacks for training control.

## Notebook

The main notebook, `Image-classification-cifar10.ipynb`, is hosted on Google Colab and can be found at the following link:
[View Notebook on Google Colab](https://colab.research.google.com/drive/1EGxv7mGw-cAkiNq7sCcV3YzoO7EB5xxZ)

## Prerequisites

To run the notebook, you need:
- Python 3.6 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib

You can install the required libraries using pip:

```bash
pip install tensorflow numpy matplotlib
```

## Usage
Download the notebook and run it in a Jupyter environment or directly on Google Colab. Ensure all the dependencies mentioned above are installed.

## Model Overview
- Two convolutional layers with ReLU activation, followed by max-pooling layers.
- Two additional convolutional layers followed by max-pooling to deepen the network.
- A fully connected layer with 512 units and dropout for regularization.
- A softmax output layer for classifying the images into one of then classes.

The model uses the Adam optimizer and categorical crossentropy as the loss function.

## Training
The model is trained with early stopping based on the accuracy reaching 90%, using a custom callback to stop training to prevent overfitting.

## Evaluation
After training, the model's performance is evaluated on a separate test set to ensure its generalization ability.

## Contributing
Contributions to this project are welcome. You can contribute in several ways:

- Improving the model's architecture or training process.
- Extending the notebook to include more visualization or analysis.
- Fixing bugs or issues.

Please create a pull request with your proposed changes.

## License
Distributed under the MIT License. See LICENSE for more information.
