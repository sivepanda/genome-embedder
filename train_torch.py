import os
import numpy as np
from gencovvec import get_coverage_vectors
from sklearn.model_selection import train_test_split 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset



device = torch.device("cuda" if torch.cuda.is_available() else cpu)

print(f'You are running {device}')

torch.cuda.memory._record_memory_history()


# Get the Coverage Vectors that define the different tracks 
create_new_coverage_vectors = False # Set to True if you would like to convert additional bed files into coverage vectors
data = get_coverage_vectors(new_coverage_vectors=create_new_coverage_vectors)

# Type cast the vectors to a float 16
data_f16= data.astype(np.float32)

# Split coverage vectors into train and test sets
x_train, x_test = train_test_split(data, test_size=0.2, random_state=50)

# Create tensors for the datasets and construct data loaders for PyTorch to train on
x_train = torch.tensor(x_train, device=device).unsqueeze(1)# .to(dtype=torch.float16)
x_test = torch.tensor(x_test, device=device).unsqueeze(1)# .to(dtype=torch.float16)
train_dataset = TensorDataset(x_train)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_datset = TensorDataset(x_train)
test_loader = DataLoader(test_datset, batch_size=50)

# Get dimensionality of layer after convolution is performed
def get_postconv_dimensionality(in_length, padding, kern_size, stride):
    dimension = ( ( in_length + ( 2 * padding ) - kern_size ) // stride ) + 1
    print(in_length, dimension)
    print(16 * dimension)
    return dimension

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

# Training function 
def train(model, dataloader, epochs=20):
    model.train()
    for epoch in range(epochs):
        for data in dataloader:
            inputs = data[0].to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).float()
            loss = criterion(outputs, inputs)
            loss.backward()
            torch.cuda.memory._dump_snapshot("snap.pickle")
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss:{loss.item()}')


input_dim = x_train.squeeze().shape[1]
print(input_dim)
model = Autoencoder(input_dim)


model.to(device)
model = model.float()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), eps=1e-4)

if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs')
    model = nn.DataParallel(model)


train(model, train_loader)


torch.save(model, os.path.join(os.getcwd(), 'models', 'model.pth'))


torch.cuda.memory._dump_snapshot("snap.pickle")
