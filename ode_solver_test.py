from torchdiffeq import odeint
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import NetworkODEModel, SparsityLoss
from dataset import NonlinearOscillatorDataset


epochs = 1
test_interval = 1
batch_size = 4
n_forecast = 5
delta = 0.1
n_samples_train = 20
n_samples_test = 10
adjacency_matrix = torch.tensor([[0, 1, 1],
                                 [1, 0, 0],
                                 [0, 1, 0]])

# Create train and test data loaders:
dataset_train = NonlinearOscillatorDataset(adjacency_matrix, n_samples=n_samples_train, n_forecast=n_forecast, delta=delta)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_test = NonlinearOscillatorDataset(adjacency_matrix, n_samples=n_samples_test, n_forecast=n_forecast, delta=delta)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Define the model:
num_nodes = adjacency_matrix.shape[0]
hidden_dim_node = 50
num_hidden_layers_node = 2
hidden_dim_coupling = 4
num_hidden_layers_coupling = 1

model = NetworkODEModel(num_nodes=num_nodes,
                        input_dim=2,
                        output_dim_nn=1,
                        hidden_dim_node=hidden_dim_node,
                        num_hidden_layers_node=num_hidden_layers_node,
                        hidden_dim_coupling=hidden_dim_coupling,
                        num_hidden_layers_coupling=num_hidden_layers_coupling)

# Optimizer and loss:
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
criterion = SparsityLoss(model, alpha=0.0)

# Train the model:
model.train()
for epoch in tqdm(range(epochs)):
    for x0, x_gt in dataloader_train:
        optimizer.zero_grad()
        x_pred = model(x0, dataset_train.t)
        loss = criterion(x_pred[:, 1:, :, :], x_gt)
        loss.backward()
        optimizer.step()

    if epoch % test_interval == 0:
        model.eval()
        with torch.no_grad():
            loss_test = 0
            for x0, x_gt in dataloader_test:
                x_pred = model(x0, dataset_train.t)
                loss_test += criterion(x_pred[:, 1:, :, :], x_gt)
            print(f'Epoch {epoch}: Test loss = {loss_test / len(dataloader_test)}')


# plot the ground-truth and predicted trajectories of each node for a sample from the test dataset in three subplots
model.eval()
x0, x_gt = next(iter(dataloader_test))
t = torch.linspace(0, 10, 50)
x_pred = model(x0, t)
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
for i in range(num_nodes):
    axs[i].plot(x_gt[0, :, i, 0].detach().numpy(), x_gt[0, :, i, 1].detach().numpy(), label=f'Node {i + 1} GT')
    axs[i].plot(x_pred[0, :, i, 0].detach().numpy(), x_pred[0, :, i, 1].detach().numpy(), label=f'Node {i + 1} Pred')
    axs[i].set_xlabel('Position')
    axs[i].set_ylabel('Velocity')
    axs[i].legend()

plt.tight_layout()
plt.show()
