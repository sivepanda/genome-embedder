import os
import numpy as np
from covvec import create_coverage_vectors 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


data = each_file()
x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)


# Define the size of the encoding
input_dim = x_train.shape[1]
encoding_dim = 70 # Dimension of the latent space

input_data = Input(shape=(input_dim,))

encoded = Dense(encoding_dim, activation='relu')(input_data)

decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_data, decoded)

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

autoencoder.save('genome_encoder_alpha.keras')
