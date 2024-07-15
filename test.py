from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import NetworkODEModel, SparsityLoss
from dataset import NonlinearOscillatorDataset, NonlinearOscillator


model_save_path = 'model_11node.pth'
test_data_file = 'data/test_11node.npz'
alpha = 0.01
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

model.load_state_dict(torch.load(model_save_path))

# plot the ground-truth and predicted trajectories of each node for a sample from the test dataset in three subplots
model.eval()
oscillator = NonlinearOscillator(dataset_test.adjacency_matrix, device=device)
x0, _ = next(iter(dataloader_test))
x0 = x0.to(device)
t = torch.linspace(0, 10, 100).to(device)
x_gt = oscillator.ode_solve(x0[0], t).to(device)
x_pred = model(x0, t)[0]
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

for i in range(3):
    axs[i].plot(x_gt[:, i, 0].cpu().detach().numpy(), x_gt[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} GT')
    axs[i].plot(x_pred[:, i, 0].cpu().detach().numpy(), x_pred[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} Pred')
    axs[i].set_xlabel('Position')
    axs[i].set_ylabel('Velocity')
    axs[i].legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 10))

for i in range(3):
    axs[i].plot(t.cpu().numpy(), x_gt[:, i, 0].cpu().detach().numpy(), label=f'Node {i + 1} GT')
    axs[i].plot(t.cpu().numpy(), x_pred[:, i, 0].cpu().detach().numpy(), label=f'Node {i + 1} Pred')
    axs[i].plot(t.cpu().numpy(), x_gt[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} GT')
    axs[i].plot(t.cpu().numpy(), x_pred[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} Pred')
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Position')
    axs[i].legend()

plt.tight_layout()
plt.show()
