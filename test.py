import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import keras

autoencoder = keras.models.load_model(os.path.join (os.getcwd(), "models", "genome_encoder_alpha.keras"))


import seaborn as sns


# Get the weights of the first layer of the encoder
weights, biases = autoencoder.layers[1].get_weights()

# Plot heatmap of the weights
plt.figure(figsize=(10, 5))
sns.heatmap(weights, cmap='viridis')
plt.xlabel("Neurons")
plt.ylabel("Input Features")
plt.title("Heatmap of Encoder Layer Weights")
plt.show()
