from torchdiffeq import odeint
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# Define the ODE system
def f(t, x):
    out = torch.empty_like(x)
    out[:, 0] = x[:, 1]
    out[:, 1] = -x[:, 0]
    return out


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(MLP, self).__init__()
        self.hidden_layers = [torch.nn.Linear(input_dim, hidden_dim)]
        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, t, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


epochs = 300
step_size = 0.1
n_forecast = 10
N_train = 1000
batch_size = 100
x_train = 2 * torch.rand(N_train, 2) - 1

# create a dataloader from x_train
dataloader = DataLoader(x_train, batch_size=batch_size, shuffle=True)
model = MLP(2, 2, 64, 2)

# Train the model using x_train as initial conditions. For each sample simulate the ODE system for 5 time units.
# The target is the solution of the ODE system at all time steps. Minimize a MSE loss over all simulated time steps using f from above to create ground truth
# trajectories. Use an Adam optimizer with a learning rate of 1e-3.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
