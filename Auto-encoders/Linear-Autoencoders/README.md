# A Simple Autoencoder


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Oo0HaTcAfkc3ZYLSh1WDEPojEOitkOFu?authuser=1)

We'll start off by building a simple autoencoder to compress the MNIST dataset. With autoencoders, we pass input data through an encoder that makes a compressed representation of the input. Then, this representation is passed through a decoder to reconstruct the input data. Generally the encoder and decoder will be built with neural networks, then trained on example data.

<img src='https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/855fb9a8f125bec59fb18f574a771107ff9ddba1/autoencoder/linear-autoencoder/notebook_ims/autoencoder_1.png' />

### Compressed Representation

A compressed representation can be great for saving and sharing any kind of data in a way that is more efficient than storing raw data. In practice, the compressed representation often holds key information about an input image and we can use it for denoising images or oher kinds of reconstruction and transformation!

<img src='https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/855fb9a8f125bec59fb18f574a771107ff9ddba1/autoencoder/linear-autoencoder/notebook_ims/denoising.png' width=60%/>

In this notebook, we'll be build a simple network architecture for the encoder and decoder. Let's get started by importing our libraries and getting the dataset.


## Linear Autoencoder

We'll train an autoencoder with these images by flattening them into 784 length vectors. The images from this dataset are already normalized such that the values are between 0 and 1. Let's start by building a simple autoencoder. The encoder and decoder should be made of **one linear layer**. The units that connect the encoder and decoder will be the _compressed representation_.

Since the images are normalized between 0 and 1, we need to use a **sigmoid activation on the output layer** to get values that match this input value range.

<img src='https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/855fb9a8f125bec59fb18f574a771107ff9ddba1/autoencoder/linear-autoencoder/notebook_ims/simple_autoencoder.png' width=50% />


#### Building the graph for the autoencoder in the cell below. 
> The input images will be flattened into 784 length vectors. The targets are the same as the inputs. 
> The encoder and decoder will be made of two linear layers, each.
> The depth dimensions should change as follows: 784 inputs > **encoding_dim** > 784 outputs.
> All layers will have ReLu activations applied except for the final output layer, which has a sigmoid activation.

**The compressed representation should be a vector with dimension `encoding_dim=32`.**


