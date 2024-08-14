from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from dissnn.model import NetworkODEModel, DissipativityLoss
from dissnn.dataset import NonlinearOscillatorDataset, NonlinearOscillator2, NonlinearOscillator3
from dissnn.dissipativity import Dissipativity, NonlinearOscillator2NodeDynamics, NonlinearOscillator3NodeDynamics, L2Gain


model_save_path = 'model_files/model_oscillator3_11node_diss_001.pth'
test_data_file = 'data/oscillator3_11node/test.npz'
# test_data_file = 'data/test_3node.npz'
batch_size = 1
device = 'cuda'
use_gt_adjacency_matrix = True
NodeDynamics = NonlinearOscillator3NodeDynamics

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

oscillator = NonlinearOscillator3(dataset_test.adjacency_matrix.to(device), device=device, **dataset_test.info)

dynamics = NodeDynamics(**dataset_test.info)
supply_rate = L2Gain()
dissipativity = Dissipativity(dynamics, supply_rate, degree=4)
dissipativity.find_storage_function()
print(f"System is dissipative with L2 gain {supply_rate.gamma_value}")
criterion_dissipativity = DissipativityLoss(dissipativity, dataset_test.adjacency_matrix, device=device).to(device)

with torch.no_grad():
    # Plot the adjacency matrix:
    # plt.imshow(model.get_adjacency_matrix().cpu().numpy())
    # plt.show()

    # for i, (x0, x_gt) in enumerate(dataloader_test):
    #     label_gt = 'Ground Truth' if i == 0 else None
    #     label_pred = 'Prediction' if i == 0 else None
    #     plt.plot(x_gt[0, :, 0, 0].detach().numpy(), x_gt[0, :, 0, 1].detach().numpy(), color='blue', label=label_gt)
    #     x0 = x0.to(device)
    #     x_gt = x_gt.to(device)
    #     x_pred = model(x0, dataset_test.t.to(device))
    #     dissipativity_loss = criterion_dissipativity(x_pred, model)
    #     plt.plot(x_pred[0, :, 0, 0].cpu().detach().numpy(), x_pred[0, :, 0, 1].cpu().detach().numpy(), color='red', label=label_pred)
    #     if dissipativity_loss > 0.0:
    #         plt.scatter(x0[0, 0, 0].cpu().detach().numpy(), x0[0, 0, 1].cpu().detach().numpy(), color='green')

    # plt.legend()
    # plt.show()

    # Simulate a trajectory and test the model on it:

    for i, (x0, x_gt) in enumerate(dataloader_test):
        x0 = x0.to(device)
        t = torch.linspace(0, 100, 1000).to(device)
        x_gt = oscillator.ode_solve(x0.squeeze(), t).unsqueeze(0)
        x_pred = model(x0, t)

        for t in range(x_gt.shape[1]):
            diss_label = 'Dissipativity Violation' if t == 0 else None
            dissipativity_loss = criterion_dissipativity(x_gt[:, t:t+1, :, :], oscillator)
            if dissipativity_loss > 0.0:
                plt.scatter(x_gt[0, t, 0, 0].cpu().detach().numpy(), x_gt[0, t, 0, 1].cpu().detach().numpy(), s=0.7, color='black', label=diss_label)

        for t in range(x_pred.shape[1]):
            diss_label = 'Dissipativity Violation' if t == 0 else None
            dissipativity_loss = criterion_dissipativity(x_pred[:, t:t+1, :, :], model)
            if dissipativity_loss > 0.0:
                plt.scatter(x_pred[0, t, 0, 0].cpu().detach().numpy(), x_pred[0, t, 0, 1].cpu().detach().numpy(), s=10, color='red', label=diss_label)
        plt.plot(x_pred[0, :, 0, 0].cpu().detach().numpy(), x_pred[0, :, 0, 1].cpu().detach().numpy(), color='green', label='Prediction')
        plt.plot(x_gt[0, :, 0, 0].cpu().detach().numpy(), x_gt[0, :, 0, 1].cpu().detach().numpy(), color='purple', label='Ground Truth')
        plt.xlabel('x1')
        plt.ylabel('x2')

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
