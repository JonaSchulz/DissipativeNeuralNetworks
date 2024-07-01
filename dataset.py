import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint
from torch.utils.data import Dataset


class NonlinearOscillator:
    def __init__(self, adjacency_matrix, a1=0.2, a2=11.0, a3=11.0, a4=1.0, mu=0.2):
        self.adjacency_matrix = adjacency_matrix
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.mu = mu

    def __call__(self, t, x):
        out = torch.empty_like(x)
        out[:, 0] = x[:, 1]
        out[:, 1] = -x[:, 0] - self.a1 * x[:, 1] * (self.a2 * x[:, 0] ** 4 - self.a3 * x[:, 0] + self.a4)
        out[:, 1] += torch.sum(self.adjacency_matrix * (x[:, 1].reshape(-1, 1) - x[:, 1].reshape(1, -1)), dim=1)
        return out

    def ode_solve(self, x0, t):
        return odeint(self, x0, t)


class NonlinearOscillatorDataset(Dataset):
    def __init__(self, adjacency_matrix, n_samples=1000, n_forecast=5, delta=0.1):
        self.adjacency_matrix = adjacency_matrix
        self.n_samples = n_samples
        self.n_forecast = n_forecast
        self.delta = delta
        self.t = torch.linspace(0, self.n_forecast * self.delta, self.n_forecast + 1)
        self.data = self.create_data()

    def create_data(self):
        num_nodes = self.adjacency_matrix.shape[0]
        oscillator = NonlinearOscillator(self.adjacency_matrix)
        x_train = 2 * torch.rand(self.n_samples, num_nodes, 2) - 1
        data = []
        for x0 in x_train:
            x = oscillator.ode_solve(x0, self.t)
            data.append(x)
        return torch.stack(data)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx, 0, :, :], self.data[idx, 1:, :, :]
