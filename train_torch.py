import numpy as np
from gencovvec import get_coverage_vectors
from sklearn.model_selection import train_test_split 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


print("You are running","cuda" if torch.cuda.is_available() else "cpu")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.memory._record_memory_history()

create_new_coverage_vectors = False # Set to True if you would like to convert additional bed files into coverage vectors
data = get_coverage_vectors(new_coverage_vectors=create_new_coverage_vectors)
data_f32 = data.astype(np.float32)

# Split coverage vectors into train and test sets
x_train, x_test = train_test_split(data_f32, test_size=0.2, random_state=50)

x_train = torch.tensor(x_train, device=device).to(dtype=torch.float32)
x_test = torch.tensor(x_test, device=device).to(dtype=torch.float32)

train_dataset = TensorDataset(x_train)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_datset = TensorDataset(x_train)
test_loader = DataLoader(test_datset, batch_size=50)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(input_dim, 500), 
                nn.ReLU(),
                nn.Linear(500, 250),
                nn.ReLU()
        )
        self.decoder = nn.Sequential(
                nn.Linear(250, input_dim),
                nn.ReLU(),
                nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


input_dim = x_train.shape[1]
print(input_dim)
model = Autoencoder(input_dim)

if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs')
    model = nn.DataParallel(model)

model.to(device)
model = model.float()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

def train(model, dataloader, epochs=12):
    model.train()
    for epoch in range(epochs):
        for data in dataloader:
            inputs = data[0].to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            torch.cuda.memory._dump_snapshot("snap.pickle")
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss:{loss.item()}')

train(model, train_loader)
torch.cuda.memory._dump_snapshot("snap.pickle")
