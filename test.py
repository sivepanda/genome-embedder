import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import keras

autoencoder = keras.models.load_model("./genome_encoder_alpha.keras")


import seaborn as sns


# Select a subset of test data for visualization
n_samples = 5
sample_indices = np.random.choice(len(x_test), n_samples)
x_test_samples = x_test[sample_indices]
x_test_pred_samples = x_test_pred[sample_indices]

# Plot original and reconstructed data
for i in range(n_samples):
    plt.figure(figsize=(10, 2))
    
    # Original data
    plt.subplot(1, 2, 1)
    plt.plot(x_test_samples[i], 'b')
    plt.title("Original")
    
    # Reconstructed data
    plt.subplot(1, 2, 2)
    plt.plot(x_test_pred_samples[i], 'r')
    plt.title("Reconstructed")
    
    plt.show()

#
## Get the weights of the first layer of the encoder
#weights, biases = autoencoder.layers[1].get_weights()
#
## Plot heatmap of the weights
#plt.figure(figsize=(10, 5))
#sns.heatmap(weights, cmap='viridis')
#plt.xlabel("Neurons")
#plt.ylabel("Input Features")
#plt.title("Heatmap of Encoder Layer Weights")
#plt.show()
