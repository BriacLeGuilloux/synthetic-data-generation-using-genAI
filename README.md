# Synthetic Data Generation with GAN

This project implements a Generative Adversarial Network (GAN) to generate synthetic application usage data.

## Project Structure

```
.
├── data/
│ └── screentime_analysis.csv # Original dataset
├── data_gen_utils.py # Utility functions for the GAN
└── explore.ipynb # Exploration and demonstration notebook
```

## Data Generation using genAI

![GAN Overview](./image/GAN_overview.png)

GANs are a type of neural network used to generate new and realistic data such as images, text, or audio. They were introduced by Ian Goodfellow in 2014.

A GAN consists of two main parts:

1. Generator

    This is a neural network whose goal is to create artificial data.
    It takes a random vector (called noise) as input and produces synthetic data.
    The generator aims to fool the discriminator by producing samples realistic enough to be mistaken for real data.

2. Discriminator

    This is another neural network that acts as a binary classifier.
    It receives either real data (from the true dataset) or generated data from the generator.
    Its task is to distinguish real data from fake (generated) data.
    It outputs a probability indicating whether the input is real or generated.

### How the GAN Works

The two networks are trained simultaneously in a zero-sum game (a kind of competition):

- The generator improves to produce increasingly realistic data to fool the discriminator.
- The discriminator improves to better detect fake data.

This adversarial training pushes both networks to improve until the generator creates data almost indistinguishable from real data.



## Features

- Synthetic data generation via GAN  
- Automatic data preprocessing  
- Distribution visualization  
- Quality evaluation of generated data

## Usage

### Using the Notebook

1. Open `explore.ipynb` in Jupyter  
2. Follow the notebook cells to:  
   - Load and preprocess the data  
   - Configure and train the GAN  
   - Generate synthetic data

## GAN Architecture

### Generator
- Input: Random noise vector (dimension 100)  
- Dense layers with LeakyReLU and BatchNormalization  
- Output: Synthetic data (3 features)

### Discriminator
- Input: Real or synthetic data  
- Dense layers with LeakyReLU  
- Output: Binary classification (real/fake)

## Configurable Parameters

- `latent_dim`: Dimension of the noise vector (default: 100)  
- `nb_epochs`: Number of training epochs (default: 10000)  
- `batch_size`: Batch size (default: 128)  
