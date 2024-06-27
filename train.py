import os
import numpy as np
from gencovvec import get_coverage_vectors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


data = get_coverage_vectors(new_coverage_vectors=True)

x_train, x_test = train_test_split(data, test_size=0.4, random_state=30)


# Define the size of the encoding
input_dim = x_train.shape[1]
# encoding_dim = int( input_dim // 2 )  # Dimension of the latent space
encoding_dim = 120 # Dimension of the latent space
print(input_dim)

input_data = Input(shape=(input_dim,))

encoded = Dense(encoding_dim, activation='relu')(input_data)
# encoding_dim = int( encoding_dim // 2 ) # Dimension of the latent space
# encoded = Dense(encoding_dim, activation='relu')(encoded)

# decoding_dim = ( encoding_dim * 2 ) 
# decoded = Dense(decoding_dim, activation='sigmoid')(encoded)
# decoding_dim = ( decoding_dim * 2 )
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_data, decoded)

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))


embeddings = autoencoder.predict( x_test )
np.save('embeddings.npy', embeddings)

print( embeddings )

# .2466
