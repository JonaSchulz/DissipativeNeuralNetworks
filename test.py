from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from dissnn.model import NetworkODEModel
from dissnn.dataset import NonlinearOscillatorDataset, NonlinearOscillator


model_save_path = 'model_files/model_oscillator1_11node.pth'
test_data_file = 'data/oscillator1_11node/test.npz'
# test_data_file = 'data/test_3node.npz'
batch_size = 1
device = 'cuda'

# Create test data loaders:
dataset_test = NonlinearOscillatorDataset(file=test_data_file)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Define the model:
num_nodes = dataset_test.num_nodes
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
                        num_hidden_layers_coupling=num_hidden_layers_coupling).to(device)

model.load(model_save_path)
model.eval()

with torch.no_grad():
    # Plot the adjacency matrix:
    plt.imshow(model.get_adjacency_matrix().cpu().numpy())
    plt.show()

    # Simulate a trajectory and test the model on it:
    oscillator = NonlinearOscillator(dataset_test.adjacency_matrix, device=device)

    x0, _ = dataset_test[0]     # Take initial condition from the test dataset
    # x0 = 2 * torch.rand(model.num_nodes, 2) - 1     # Random initial condition
    x0 = x0.unsqueeze(0).to(device)
    t = torch.linspace(0, 10, 100).to(device)
    x_gt = oscillator.ode_solve(x0[0], t).to(device)
    x_pred = model(x0, t)[0]

# Plot the ground-truth and predicted trajectories of each node:
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

for i in range(3):
    axs[i].plot(x_gt[:, i, 0].cpu().detach().numpy(), x_gt[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} GT')
    axs[i].plot(x_pred[:, i, 0].cpu().detach().numpy(), x_pred[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} Pred')
    axs[i].set_xlabel('Position')
    axs[i].set_ylabel('Velocity')
    axs[i].legend()

plt.tight_layout()
plt.show()

# Plot the ground-truth and predicted state evolution over time of each node for a sample from the test dataset:
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

for i in range(3):
    axs[i].plot(t.cpu().numpy(), x_gt[:, i, 0].cpu().detach().numpy(), label=f'Node {i + 1} GT Position')
    axs[i].plot(t.cpu().numpy(), x_pred[:, i, 0].cpu().detach().numpy(), label=f'Node {i + 1} Pred Position')
    axs[i].plot(t.cpu().numpy(), x_gt[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} GT Velocity')
    axs[i].plot(t.cpu().numpy(), x_pred[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} Pred Velocity')
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Position/Velocity')
    axs[i].legend()

plt.tight_layout()
plt.show()
