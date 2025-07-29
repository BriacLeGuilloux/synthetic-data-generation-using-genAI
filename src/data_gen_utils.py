"""
Utility functions for synthetic data generation using GANs
This module provides the core functionality for generating synthetic data using Generative Adversarial Networks.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def build_generator(latent_dim: int) -> Sequential:
    """
    Build the generator model for the GAN
    
    Args:
        latent_dim: Dimension of the latent space (noise input)
        
    Returns:
        Sequential: Compiled generator model
    """
    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.01),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(alpha=0.01),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.01),
        BatchNormalization(momentum=0.8),
        Dense(3, activation='sigmoid')  # output layer for generating features
    ])
    return model

def build_discriminator() -> Sequential:
    """
    Build the discriminator model for the GAN
    
    Returns:
        Sequential: Compiled discriminator model
    """
    model = Sequential([
        Dense(512, input_shape=(3,)),
        LeakyReLU(alpha=0.01),
        Dense(256),
        LeakyReLU(alpha=0.01),
        Dense(128),
        LeakyReLU(alpha=0.01),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', 
                 optimizer=Adam(), 
                 metrics=['accuracy'])
    return model

def build_gan(generator: Sequential, discriminator: Sequential) -> Sequential:
    """
    Combine generator and discriminator into a GAN model
    
    Args:
        generator: The generator model
        discriminator: The discriminator model
        
    Returns:
        Sequential: Compiled GAN model
    """
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model

def train_gan(gan: Sequential, 
             generator: Sequential, 
             discriminator: Sequential, 
             data: np.ndarray, 
             nb_epochs: int = 10000, 
             batch_size: int = 128, 
             latent_dim: int = 100) -> None:
    """
    Train the GAN model
    
    Args:
        gan: The combined GAN model
        generator: The generator model
        discriminator: The discriminator model
        data: Training data
        nb_epochs: Number of training epochs
        batch_size: Size of training batches
        latent_dim: Dimension of latent space
    """
    for epoch in range(nb_epochs):
        # Train discriminator on real data
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        
        # Generate and train on fake data
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)
        
        # Labels for training
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_labels)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: D Loss: {0.5 * np.add(d_loss_real, d_loss_fake)}, G Loss: {g_loss}")

def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Preprocess the input data for GAN training
    
    Args:
        data: Input DataFrame
        
    Returns:
        Tuple containing:
        - normalized_data: Preprocessed numpy array
        - scaler: Fitted MinMaxScaler for inverse transformation
    """
    data_gan = data.drop(columns=['Date', 'App'])
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data_gan)
    return normalized_data, scaler

def generate_synthetic_data(generator: Sequential, 
                          scaler: MinMaxScaler, 
                          n_samples: int, 
                          latent_dim: int, 
                          feature_names: list) -> pd.DataFrame:
    """
    Generate synthetic data using the trained generator
    
    Args:
        generator: Trained generator model
        scaler: Fitted MinMaxScaler
        n_samples: Number of samples to generate
        latent_dim: Dimension of latent space
        feature_names: Names of the features
        
    Returns:
        DataFrame containing generated data
    """
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    generated_data = generator.predict(noise)
    generated_data_rescaled = scaler.inverse_transform(generated_data)
    return pd.DataFrame(generated_data_rescaled, columns=feature_names)
