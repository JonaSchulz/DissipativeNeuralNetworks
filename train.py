from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow

from dissnn.model import NetworkODEModel, SparsityLoss, DissipativityLoss
from dissnn.dataset import NonlinearOscillatorDataset, NonlinearOscillator
from dissnn.dissipativity import Dissipativity, NodeDynamics, L2Gain


model_save_path = 'model_files/model_oscillator2_11node_sic.pth'
train_data_file = 'data/oscillator2_11node_sic/train.npz'
test_data_file = 'data/oscillator2_11node_sic/test.npz'
epochs = 200
test_interval = 20
batch_size = 32
device = 'cuda'
sparsity_weight = 0.0
dissipativity_weight = 0.0

# Create train and test data loaders:
dataset_train = NonlinearOscillatorDataset(file=train_data_file)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_test = NonlinearOscillatorDataset(file=test_data_file)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Dissipativity:
dynamics = NodeDynamics(alpha=0.1, beta=0.1, k=0.1)
supply_rate = L2Gain()
dissipativity = Dissipativity(dynamics, supply_rate, degree=4)
dissipativity.find_storage_function()
print(f"System is dissipative with L2 gain {supply_rate.gamma_value}")

# Define the model:
num_nodes = dataset_train.num_nodes
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

# Optimizer and loss:
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
criterion = SparsityLoss(model, alpha=sparsity_weight).to(device)
criterion_dissipativity = DissipativityLoss(dissipativity, dataset_train.adjacency_matrix, device=device).to(device)

# Train the model:
with mlflow.start_run():
    params = {
        'epochs': epochs,
        'batch_size': batch_size,
        'sparsity_weight': sparsity_weight,
        'dissipativity_weight': dissipativity_weight
    }
    params.update(dataset_train.info)
    mlflow.log_params(params)

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

        scheduler.step()
        
        if epoch % test_interval == 0:
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

                #scheduler.step(val_loss)
                print(f'Epoch {epoch}: Test loss = {val_loss / len(dataloader_test)}')

    torch.save(model.state_dict(), model_save_path)
    mlflow.pytorch.log_model(model, "final_model")

# plot the ground-truth and predicted trajectories of each node for a sample from the test dataset in three subplots
model.eval()
oscillator = NonlinearOscillator(dataset_test.adjacency_matrix, device=device)
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
