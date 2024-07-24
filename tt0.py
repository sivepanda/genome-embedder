import numpy as np
from gencovvec import get_coverage_vectors
from sklearn.model_selection import train_test_split 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):

    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        print("Initializing autoencoder")
        in_c = 1
        out = 32
        kern = 200 
        stri = 4
        conv_dim = out * ((input_dim - kern) // stri + 1)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_c, out, kern, stride=stri, padding=0),
            nn.ReLU(), 
            nn.Flatten(),
            nn.Linear((conv_dim), 1000), 
            nn.ReLU(),
            nn.Linear(500, 250),
            # nn.ReLU(),
            # nn.Linear(500, 250),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, (conv_dim)),
            nn.ReLU(),
            nn.Unflatten(in_c, (out, ((input_dim - kern) // stri + 1))),
            nn.ConvTranspose1d(in_c, out, kern, stride=stri, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.squeeze(1)

def train(model, dataloader, epochs=100):
    model.train()
    for epoch in range(epochs):
        for data in dataloader:
            inputs = data[0].to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).float()
            loss = criterion(outputs, inputs)
            loss.backward()
            # torch.cuda.memory._dump_snapshot("snap.pickle")
            optimizer.step()
        print(f'Epoch {epoch+1}\tLoss:{loss.item()}\n')


def parallelize(model):
    print("Initializing device setup...")
    if torch.cuda.device_count() > 1:
        print(f'Using CUDA Devices.')
        print(f'Using {torch.cuda.device_count()} GPUs in parallel')
        model = nn.DataParallel(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.cuda.memory._record_memory_history()


create_new_coverage_vectors = False # Set to True if you would like to convert additional bed files into coverage vectors
data = get_coverage_vectors(new_coverage_vectors=create_new_coverage_vectors)
data_f32 = data.astype(np.float32)

# Split coverage vectors into train and test sets
x_train, x_test = train_test_split(data, test_size=0.2, random_state=50)

x_train = torch.tensor(x_train, device=device)#.to(dtype=torch.float32)
x_test = torch.tensor(x_test, device=device)#.to(dtype=torch.float32)

train_dataset = TensorDataset(x_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_datset = TensorDataset(x_train)
test_loader = DataLoader(test_datset, batch_size=128)

input_dim = x_train.shape[1]
print("Input Dimension:", input_dim)
model = Autoencoder(input_dim)
parallelize(model)


print("\n\n")

model.to(device)
model = model.float()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


train(model, train_loader)
# torch.cuda.memory._dump_snapshot("snap.pickle")
