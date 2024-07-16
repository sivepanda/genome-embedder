import os
import gc
import numpy as np
from gencovvec import get_coverage_vectors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l1

 

# Options

# Set to True if you would like to convert additional bed files into coverage vectors
create_new_coverage_vectors = False 

# Proprtion of the total dataset to be used to test (1 - training proprtion)
train_test_proportion = 0.2

# Size of batch that is sent to be trained at one time
batch_size = 1

# Environment configuration

# Set memory growth to true to allow dynamic memory allocation as needs potentially grow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=30000)]) # 4GB limit
    except RuntimeError as e:
        print(e)



# Create a "strategy" of allowing tasks to be distributed across GPUs
strategy = tf.distribute.MirroredStrategy()
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)



print("Getting coverage vectors...")
# Get coverage vectors
data = get_coverage_vectors(new_coverage_vectors=create_new_coverage_vectors)
data_f32 = data.astype(np.float16)

print("Creating training and test datasets...")
dataset = tf.data.Dataset.from_tensor_slices(data)
shuffled_dataset = dataset.shuffle(buffer_size=len(data), reshuffle_each_iteration=True)

# Split coverage vectors into train and test sets
test_size = int(train_test_proportion * len(data))
train_size = len(data) - test_size

print(train_size, test_size)

x_train_ds = shuffled_dataset.take(train_size).batch(batch_size)#.prefetch(tf.data.experimental.AUTOTUNE).repeat()
x_test_ds = shuffled_dataset.take(test_size).batch(batch_size)#.prefetch(tf.data.experimental.AUTOTUNE).repeat()
print("k")

def map_fn(n):
    return (n, n)

x_train_ds = x_train_ds.map(map_fn)
x_test_ds = x_test_ds.map(map_fn)

gc.collect()


# x_train, x_test = train_test_split(data_f32, test_size=train_test_proportion, random_state=50)

# Define the size of the encoding
# input_dim = x_train.shape[1]

# Sets dimension of latent space
encoding_dim = 200#int( input_dim // 2 ) # too large?


with strategy.scope():
    # input_dim = next(iter(x_train_ds)).shape[1]
    # print(next(iter(x_train_ds)))

    # input_dim = next(iter(x_train_ds))
    # print(type(input_dim))
    
    input_dim = next(iter(x_train_ds))[0].shape[1]
    print('Size of input data', input_dim)
    # print("Training dataset shape:", x_train.shape)
    # print("Test dataset shape:", x_test.shape)

    input_data = Input(shape=(input_dim,))

    # encoded = Dense(encoding_dim, activation='relu')(input_data)
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=l1(10e-4))(input_data)
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
    autoencoder.fit(x_train_ds, epochs=12, steps_per_epoch= train_size // batch_size, validation_data=x_test_ds, validation_steps=test_size // batch_size )

# encoder.save( './encoder.keras' )

embeddings = encoder.predict( x_test )
np.save('embeddings.npy', embeddings)
