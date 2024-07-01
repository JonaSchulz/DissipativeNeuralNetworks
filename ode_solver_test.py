from torchdiffeq import odeint
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import NetworkODEModel
from dataset import NonlinearOscillatorDataset


epochs = 1
batch_size = 32
adjacency_matrix = torch.tensor([[0, 1, 1],
                                 [1, 0, 0],
                                 [0, 1, 0]])

# Create train and test data loaders:
dataset_train = NonlinearOscillatorDataset(adjacency_matrix)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_test = NonlinearOscillatorDataset(adjacency_matrix, n_samples=100)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Define the model:
num_nodes = adjacency_matrix.shape[0]
hidden_dim_node = 50
num_hidden_layers_node = 2
hidden_dim_coupling = 4
num_hidden_layers_coupling = 4

model = NetworkODEModel(num_nodes=num_nodes,
                        input_dim=2,
                        output_dim=1,
                        hidden_dim_node=hidden_dim_node,
                        num_hidden_layers_node=num_hidden_layers_node,
                        hidden_dim_coupling=hidden_dim_coupling,
                        num_hidden_layers_coupling=num_hidden_layers_coupling)

optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
criterion = torch.nn.MSELoss()
model.train()
for epoch in tqdm(range(epochs)):
    for i, x0 in enumerate(dataloader):
        optimizer.zero_grad()
        t = torch.arange(0, n_forecast * step_size, step_size)
        x_pred = odeint(model, x0, t)
        x_gt = odeint(f, x0, t)
        loss = criterion(x_pred, x_gt)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    t = torch.linspace(0, 25, 1000)
    x0 = torch.tensor([[1., 0.]])
    x_gt = odeint(f, x0, t).squeeze()
    x_pred = odeint(model, x0, t).squeeze()

plt.plot(t, x_gt[:, 0], label='gt')
plt.plot(t, x_pred[:, 0], label='pred')
plt.legend()
plt.show()
