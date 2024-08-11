from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from dissnn.model import NetworkODEModel
from dissnn.dataset import NonlinearOscillatorDataset, NonlinearOscillator


model_save_path = 'model_files/model_oscillator2_11node_3_sic.pth'
test_data_file = 'data/oscillator2_11node_3_sic/train.npz'
# test_data_file = 'data/test_3node.npz'
batch_size = 1
device = 'cuda'
use_gt_adjacency_matrix = False

# Create test data loaders:
dataset_test = NonlinearOscillatorDataset(file=test_data_file)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Define the model:
num_nodes = dataset_test.num_nodes
hidden_dim_node = 50
num_hidden_layers_node = 2
hidden_dim_coupling = 4
num_hidden_layers_coupling = 1
adjacency_matrix = dataset_test.adjacency_matrix.to(float).to(device) if use_gt_adjacency_matrix else None

model = NetworkODEModel(num_nodes=num_nodes,
                        input_dim=2,
                        output_dim_nn=1,
                        hidden_dim_node=hidden_dim_node,
                        num_hidden_layers_node=num_hidden_layers_node,
                        hidden_dim_coupling=hidden_dim_coupling,
                        num_hidden_layers_coupling=num_hidden_layers_coupling,
                        adjacency_matrix=adjacency_matrix).to(device)

model.load(model_save_path)
model.eval()

with torch.no_grad():
    # Plot the adjacency matrix:
    plt.imshow(model.get_adjacency_matrix().cpu().numpy())
    plt.show()

    for i, (x0, x_gt) in enumerate(dataloader_test):
        label_gt = 'Ground Truth' if i == 0 else None
        label_pred = 'Prediction' if i == 0 else None
        plt.plot(x_gt[0, :, 0, 0].detach().numpy(), x_gt[0, :, 0, 1].detach().numpy(), color='blue', label=label_gt)
        x0 = x0.to(device)
        x_gt = x_gt.to(device)
        x_pred = model(x0, dataset_test.t.to(device))
        plt.plot(x_pred[0, :, 0, 0].cpu().detach().numpy(), x_pred[0, :, 0, 1].cpu().detach().numpy(), color='red', label=label_pred)

    # Simulate a trajectory and test the model on it:
    oscillator = NonlinearOscillator(dataset_test.adjacency_matrix, device=device)

    for i, (x0, x_gt) in enumerate(dataloader_test):
        x0 = x0.to(device)
        t = torch.linspace(0, (dataset_test.n_samples + dataset_test.n_forecast) * dataset_test.delta, (dataset_test.n_samples + dataset_test.n_forecast)).to(device)
        x_pred = model(x0, t)
        plt.plot(x_pred[0, :, 0, 0].cpu().detach().numpy(), x_pred[0, :, 0, 1].cpu().detach().numpy(), color='green', label="Pred From Start")
        break

    plt.legend()
    plt.show()

exit()
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
