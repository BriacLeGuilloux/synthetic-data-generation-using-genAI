import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('./data/Screentime-App-Details.csv')

print(data.head())


# drop unnecessary columns
data_gan = data.drop(columns=['Date', 'App'])

# initialize a MinMaxScaler to normalize the data between 0 and 1
scaler = MinMaxScaler()

# normalize the data
normalized_data = scaler.fit_transform(data_gan)

# convert back to a DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=data_gan.columns)

normalized_df.head()


latent_dim = 100  # size of the random noise vector

latent_dim = 100  # latent space dimension (size of the random noise input)

def build_generator(latent_dim):
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
        Dense(3, activation='sigmoid')  # output layer for generating 3 features
    ])
    return model

# create the generator
generator = build_generator(latent_dim)
generator.summary()


# generate random noise for 1000 samples
noise = np.random.normal(0, 1, (1000, latent_dim))

# generate synthetic data using the generator
generated_data = generator.predict(noise)

# display the generated data
generated_data[:5]  # show first 5 samples

def build_discriminator():
    model = Sequential([
        Dense(512, input_shape=(3,)),
        LeakyReLU(alpha=0.01),
        Dense(256),
        LeakyReLU(alpha=0.01),
        Dense(128),
        LeakyReLU(alpha=0.01),
        Dense(1, activation='sigmoid')  # output: 1 neuron for real/fake classification
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# create the discriminator
discriminator = build_discriminator()
discriminator.summary()


def build_gan(generator, discriminator):
    # freeze the discriminator’s weights while training the generator
    discriminator.trainable = False

    model = Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model

# create the GAN
gan = build_gan(generator, discriminator)
gan.summary()