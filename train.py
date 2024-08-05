from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import NetworkODEModel, SparsityLoss
from dataset import NonlinearOscillatorDataset, NonlinearOscillator


model_save_path = 'model_11node_single_initial_condition.pth'
train_data_file = 'data/train_11node_single_initial_condition.npz'
test_data_file = 'data/test_11node_single_initial_condition.npz'
epochs = 1000
alpha = 0.01
test_interval = 10
batch_size = 32
device = 'cuda'

# Create train and test data loaders:
dataset_train = NonlinearOscillatorDataset(file=train_data_file)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_test = NonlinearOscillatorDataset(file=test_data_file)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
criterion = SparsityLoss(model, alpha=0.0).to(device)

# Train the model:
model.train()
best_test_loss = float('inf')
for epoch in tqdm(range(epochs)):
    if epoch >= epochs // 2:
        criterion.alpha = alpha

    for x0, x_gt in dataloader_train:
        x0 = x0.to(device)
        x_gt = x_gt.to(device)

        optimizer.zero_grad()
        x_pred = model(x0, dataset_train.t.to(device))
        loss = criterion(x_pred[:, 1:, :, :], x_gt)
        loss.backward()
        optimizer.step()

    if epoch % test_interval == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x0, x_gt in dataloader_test:
                x0 = x0.to(device)
                x_gt = x_gt.to(device)

                x_pred = model(x0, dataset_train.t.to(device))
                val_loss += criterion(x_pred[:, 1:, :, :], x_gt)

            if val_loss < best_test_loss:
                best_test_loss = val_loss
                torch.save(model.state_dict(), model_save_path)

            scheduler.step(val_loss)
            print(f'Epoch {epoch}: Test loss = {val_loss / len(dataloader_test)}')

# torch.save(model.state_dict(), model_save_path)

# plot the ground-truth and predicted trajectories of each node for a sample from the test dataset in three subplots
model.eval()
oscillator = NonlinearOscillator(dataset_test.adjacency_matrix, device=device)
x0, _ = next(iter(dataloader_test))
x0 = x0.to(device)
t = torch.linspace(0, 5, 10).to(device)
x_gt = oscillator.ode_solve(x0[0], t).to(device)
x_pred = model(x0, t)[0]
fig, axs = plt.subplots(11, 1, figsize=(10, 10))
for i in range(num_nodes):
    axs[i].plot(x_gt[:, i, 0].cpu().detach().numpy(), x_gt[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} GT')
    axs[i].plot(x_pred[:, i, 0].cpu().detach().numpy(), x_pred[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} Pred')
    axs[i].set_xlabel('Position')
    axs[i].set_ylabel('Velocity')
    axs[i].legend()

plt.tight_layout()
plt.show()
