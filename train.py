import os
import numpy as np
from gencovvec import get_coverage_vectors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('hey')
    except RuntimeError as e:
        print(e)
strategy = tf.distribute.MirroredStrategy()



# Get coverage vectors
create_new_coverage_vectors = False # Set to True if you would like to convert additional bed files into coverage vectors
data = get_coverage_vectors(new_coverage_vectors=create_new_coverage_vectors)
data_f32 = data.astype(np.float16)

# Split coverage vectors into train and test sets
x_train, x_test = train_test_split(data_f32, test_size=0.2, random_state=50)

with strategy.scope():

    print(x_train.shape)
    print(x_test.shape)
    # Define the size of the encoding
    input_dim = x_train.shape[1]

    # Sets dimension of latent space

    encoding_dim = 250 #int( input_dim // 2 ) # too large?
    print('Size of input data', input_dim)


    input_data = Input(shape=(input_dim,))

    encoded = Dense(1000, activation='relu')(input_data)
    # Set next column to have decrease dimensionality by half
    # encoding_dim = int( encoding_dim // 2 ) 
    # encoded = Dense(encoding_dim, activation='relu')(encoded)

    encoder = Model(input_data, encoded) # Save model

    # Set next colum to twice the previous dimensionality before returning to the initial dimensionality
    # decoding_dim = ( encoding_dim * 2 ) 
    # decoded = Dense(decoding_dim, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_data, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x_train, x_train,
            epochs=12,
            batch_size=50,
            shuffle=True,
            validation_data=(x_test, x_test))

encoder.save( './encoder.keras' )

embeddings = encoder.predict( x_test )
np.save('embeddings.npy', embeddings)
