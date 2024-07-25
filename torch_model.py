import os
import numpy as np
import torch
import torch.nn as nn

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=100, stride=20, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear( ( get_postconv_dimensionality(input_dim, 0, 100, 20) * 16), 250 ),
            nn.ReLU()
        )

        self.encoder = nn.Sequential(
            nn.Linear(250, 120),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(120, 250),
            nn.ReLU(), 
        )
        
        self.upsample = nn.Sequential(
            nn.Linear( 250, (16 * get_postconv_dimensionality(input_dim, 0, 100, 20) ) ),
            nn.ReLU(),
            nn.Unflatten(1, (16, get_postconv_dimensionality(input_dim, 0, 100, 20))),
            nn.ConvTranspose1d(16, 1, kernel_size=100, stride=20, padding=0, output_padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        downsampled = self.downsample(x)
        encoded = self.encoder(downsampled)
        decoded = self.decoder(encoded)
        upsampled = self.upsample(decoded)
        return upsampled 
