from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
import tempfile
import os

from dissnn.model import NetworkODEModel, SparsityLoss, DissipativityLoss
from dissnn.dataset import NonlinearOscillatorDataset, NonlinearOscillator2
from dissnn.dissipativity import Dissipativity, NonlinearOscillator2NodeDynamics, L2Gain, Passivity, DissipativityPendulum


# How to use:
# - select train and test data files
# - choose whether to use the ground truth adjacency matrix or learn it
# - select correct ground truth node dynamics class

model_save_path = 'model_files/model_pendulum_3node.pth'
train_data_file = 'data/pendulum_3node/train.npz'
test_data_file = 'data/pendulum_3node/test.npz'
epochs = 20
test_interval = 5
batch_size = 64
device = 'cuda'
sparsity_weight = 0.0
dissipativity_weight = 0.0
use_gt_adjacency_matrix = True
storage_function_degree = 4
NodeDynamics = NonlinearOscillator2NodeDynamics

# Create train and test data loaders:
dataset_train = NonlinearOscillatorDataset(file=train_data_file)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_test = NonlinearOscillatorDataset(file=test_data_file)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Dissipativity:
# dynamics = NodeDynamics(**dataset_train.info)
# supply_rate = L2Gain()
# dissipativity = Dissipativity(dynamics, supply_rate, degree=storage_function_degree)
# dissipativity.find_storage_function()
# print(f"System is dissipative with L2 gain {supply_rate.gamma_value}")

supply_rate = Passivity()
dissipativity = DissipativityPendulum(**dataset_train.info)

# Define the model:
num_nodes = dataset_train.num_nodes
hidden_dim_node = 50
num_hidden_layers_node = 4
hidden_dim_coupling = 50
num_hidden_layers_coupling = 4
adjacency_matrix = dataset_train.adjacency_matrix.to(float).to(device) if use_gt_adjacency_matrix else None

model = NetworkODEModel(num_nodes=num_nodes,
                        input_dim=2,
                        output_dim_nn=1,
                        hidden_dim_node=hidden_dim_node,
                        num_hidden_layers_node=num_hidden_layers_node,
                        hidden_dim_coupling=hidden_dim_coupling,
                        num_hidden_layers_coupling=num_hidden_layers_coupling,
                        adjacency_matrix=adjacency_matrix).to(device)

# Optimizer and loss:
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = SparsityLoss(model, alpha=sparsity_weight).to(device)
criterion_dissipativity = DissipativityLoss(dissipativity, dataset_train.adjacency_matrix, device=device).to(device)

# Train the model:
with mlflow.start_run():
    params = {
        'epochs': epochs,
        'batch_size': batch_size,
        'sparsity_weight': sparsity_weight,
        'dissipativity_weight': dissipativity_weight,
        'use_gt_adjacency_matrix': use_gt_adjacency_matrix,
        'dataset_train': train_data_file,
        'dataset_test': test_data_file,
        'storage_function_degree': storage_function_degree,
        'hidden_dim_node': hidden_dim_node,
        'num_hidden_layers_node': num_hidden_layers_node,
        'hidden_dim_coupling': hidden_dim_coupling,
        'num_hidden_layers_coupling': num_hidden_layers_coupling,
    }
    params.update(dataset_train.info)
    if isinstance(dissipativity, Dissipativity):
        params.update({'storage_coefficients': [str(i) for i in dissipativity.coefficients],
                       'storage_coefficient_values': dissipativity.coefficient_values})
    mlflow.log_params(params)
    if isinstance(supply_rate, L2Gain):
        mlflow.log_param('gamma_value', dissipativity.supply_rate.gamma_value)

    model.train()
    best_test_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        if epoch >= epochs // 2:
            criterion.alpha = sparsity_weight

        for x0, x_gt in dataloader_train:
            x0 = x0.to(device)
            x_gt = x_gt.to(device)

            optimizer.zero_grad()
            x_pred = model(x0, dataset_train.t.to(device))
            sparsity_loss = criterion(x_pred[:, 1:, :, :], x_gt)
            dissipativity_loss = criterion_dissipativity(x_pred, model)
            loss = sparsity_loss + dissipativity_weight * dissipativity_loss
            loss.backward()
            optimizer.step()

            mlflow.log_metric("train/mse_loss", sparsity_loss.item())
            mlflow.log_metric("train/dissipativity_loss", dissipativity_loss.item())

            del x_pred, sparsity_loss, dissipativity_loss, loss
            torch.cuda.empty_cache()

        scheduler.step()
        
        if epoch % test_interval == 0:
            # Log the adjacency matrix as an image to MLflow:
            adjacency_matrix = model.get_adjacency_matrix().detach().cpu().numpy()
            plt.imshow(adjacency_matrix)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                plt.savefig(temp.name)
                temp.close()
                # Log the temporary file as an artifact to MLflow
                mlflow.log_artifact(temp.name, "adjacency_matrix_images")
                # Remove the temporary file
                os.remove(temp.name)

            plt.close()

            model.eval()
            with torch.no_grad():
                val_loss = 0
                for x0, x_gt in dataloader_test:
                    x0 = x0.to(device)
                    x_gt = x_gt.to(device)

                    x_pred = model(x0, dataset_train.t.to(device))
                    sparsity_loss = criterion(x_pred[:, 1:, :, :], x_gt)
                    dissipativity_loss = criterion_dissipativity(x_pred, model)
                    loss = sparsity_loss + dissipativity_weight * dissipativity_loss

                    mlflow.log_metric("test/mse_loss", sparsity_loss.item())
                    mlflow.log_metric("test/dissipativity_loss", dissipativity_loss.item())

                    val_loss += loss

                if val_loss < best_test_loss:
                    best_test_loss = val_loss
                    torch.save(model.state_dict(), model_save_path)
                    mlflow.pytorch.log_model(model, "best_model")

                # scheduler.step(val_loss)
                print(f'Epoch {epoch}: Test loss = {val_loss / len(dataloader_test)}')

            model.train()

    torch.save(model.state_dict(), model_save_path)
    mlflow.pytorch.log_model(model, "final_model")

# plot the ground-truth and predicted trajectories of each node for a sample from the test dataset in three subplots
model.eval()
oscillator = NonlinearOscillator2(dataset_test.adjacency_matrix.to(device), device=device, **dataset_test.info)
x0, _ = next(iter(dataloader_test))
x0 = x0.to(device)
t = torch.linspace(0, 5, 100).to(device)
x_gt = oscillator.ode_solve(x0[0], t).to(device)
x_pred = model(x0, t)[0]
fig, axs = plt.subplots(5, 1, figsize=(10, 10))
for i in range(5):
    axs[i].plot(x_gt[:, i, 0].cpu().detach().numpy(), x_gt[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} GT')
    axs[i].plot(x_pred[:, i, 0].cpu().detach().numpy(), x_pred[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} Pred')
    axs[i].set_xlabel('Position')
    axs[i].set_ylabel('Velocity')
    axs[i].legend()

plt.tight_layout()
plt.show()
