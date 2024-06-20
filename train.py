import os
import numpy as np
from covvec import create_coverage_vectors 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def each_file():
    # Specify the directory path
    directory_path = os.path.join(os.getcwd(), "tracks")
    
    all_data = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            temp = create_coverage_vectors( feature_file = filename )
            print(len(temp))
            all_data.append(temp)
        else:
            print(f"Found directory: {filename}")

    return np.array( all_data )

data = each_file()
x_train, x_test = train_test_split(data, test_size=0.2, random_state=42)


# Define the size of the encoding
input_dim = x_train.shape[1]
encoding_dim = 32  # Dimension of the latent space

input_data = Input(shape=(input_dim,))

encoded = Dense(encoding_dim, activation='relu')(input_data)

decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_data, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))



# Calculate reconstruction error for test set
x_test_pred = autoencoder.predict(x_test)
reconstruction_error = np.mean(np.square(x_test - x_test_pred), axis=1)

for r in reconstruction_error:
    print(r)

# Plot histogram of reconstruction errors
plt.hist(reconstruction_error, bins=4)
plt.xlabel("Reconstruction error")
plt.ylabel("Number of samples")
plt.title("Reconstruction error histogram")
plt.show()
