import os
import numpy as np
from gencovvec import get_coverage_vectors
from sklearn.model_selection import train_test_split 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP 

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1000), 
                nn.ReLU(),
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.Linear(500, 250),
                nn.ReLU()
        )
        self.decoder = nn.Sequential(
                nn.Linear(250, 500),
                nn.ReLU(),
                nn.Linear(500, 1000),
                nn.ReLU(),
                nn.Linear(1000, input_dim),
                nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_model(model, dataloader, epochs=100):
    model.train()
    for epoch in range(epochs):
        for data in dataloader:
            inputs = data[0].to(rank).float()
            optimizer.zero_grad()
            outputs = model(inputs).float()
            loss = criterion(outputs, inputs)
            loss.backward()
            torch.cuda.memory._dump_snapshot("snap.pickle")
            optimizer.step()
        print(f'Epoch {epoch+1}\tLoss:{loss.item()}\n')

def parallelize(model):
    print("Initializing device setup...")
    if torch.cuda.device_count() > 1:
        print(f'Using CUDA Devices.')
        print(f'Using {torch.cuda.device_count()} GPUs in parallel')
        model = nn.DataParallel(model)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

def dist_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '65335'
    print("Setting up distributed computing...")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Setup complete.")
    torch.cuda.set_device(rank)

def cleanup():
    print("Cleaning up processes...")
    dist.distroy_process_group()
    print("Cleanup complete.")



def train(rank, world_size):
    dist_setup(rank, world_size)
    create_new_coverage_vectors = False # Set to True if you would like to convert additional bed files into coverage vectors

    data = get_coverage_vectors(new_coverage_vectors=create_new_coverage_vectors)
    data_f32 = data.astype(np.float32)

    # Split coverage vectors into train and test sets
    x_train, x_test = train_test_split(data, test_size=0.2, random_state=50)

    x_train = torch.tensor(x_train, device=rank)#.to(dtype=torch.float32)
    x_test = torch.tensor(x_test, device=rank)#.to(dtype=torch.float32)

    train_dataset = TensorDataset(x_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_datset = TensorDataset(x_train)
    test_loader = DataLoader(test_datset, batch_size=128)

    input_dim = x_train.shape[1]
    print("Input Dimension:", input_dim)
    model = Autoencoder(input_dim)
    # parallelize(model)

    print("\n\n")

    # model.to(device)
    model = model.float()
    model = DDP(model, device_ids=[rank])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader)
    
    cleanup()

# torch.cuda.memory._record_memory_history()


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)

# torch.cuda.memory._dump_snapshot("snap.pickle")
