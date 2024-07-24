import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint
from torch.utils.data import Dataset
import json
from functools import partial


class NonlinearOscillator:
    def __init__(self, adjacency_matrix, a1=0.2, a2=11.0, a3=11.0, a4=1.0, mu=0.2, device='cpu'):
        self.adjacency_matrix = adjacency_matrix.to(device)
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.mu = mu
        self.device = device

    def __call__(self, t, x):
        out = torch.empty_like(x).to(self.device)
        out[:, 0] = x[:, 1]
        out[:, 1] = -x[:, 0] - self.a1 * x[:, 1] * (self.a2 * x[:, 0] ** 4 - self.a3 * x[:, 0] + self.a4)
        out[:, 1] += torch.sum(self.adjacency_matrix * (x[:, 1].reshape(-1, 1) - x[:, 1].reshape(1, -1)), dim=1)
        return out

    def ode_solve(self, x0, t):
        return odeint(self, x0, t)


class KuramotoOscillator:
    def __init__(self, adjacency_matrix, omega=1, device='cpu', k=1.0):
        self.adjacency_matrix = adjacency_matrix.to(device)
        self.device = device
        self.omega = omega
        self.k = k

    def __call__(self, t, x):
        out = torch.empty_like(x).to(self.device)
        out[:, 0] = self.omega + self.k * torch.mean(self.adjacency_matrix * torch.sin(x[:, 0].reshape(1, -1) - x[:, 0].reshape(-1, 1)), dim=1)
        return out

    def ode_solve(self, x0, t):
        return odeint(self, x0, t)


class HarmonicOscillator:
    def __init__(self, adjacency_matrix, c=1, m=1, k=1):
        self.adjacency_matrix = adjacency_matrix
        self.c = c
        self.m = m
        self.k = k

    def __call__(self, t, x):
        out = torch.empty_like(x)
        out[:, 0] = x[:, 1]
        out[:, 1] = -self.c / self.m * x[:, 1] - self.k / self.m * x[:, 0]
        out[:, 1] -= torch.sum(self.adjacency_matrix * (x[:, 1].reshape(-1, 1) - x[:, 1].reshape(1, -1)), dim=1)
        return out

    def ode_solve(self, x0, t):
        return odeint(self, x0, t)


class RingNetwork:
    def __init__(self, num_nodes, alpha=1.0, beta=1.0):
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.beta = beta
        self.time_step = 0
        self.u = None

    def __call__(self, t, x):
        out = torch.empty_like(x)
        out[:, 0] = -self.alpha * x[:, 0] ** 3 + self.beta * (x[:, 0].roll(1, 0) - x[:, 0])
        if self.u is not None:
            out[0, 0] += self.u(t)
        return out

    def ode_solve(self, x0, t, u=None):
        self.u = u
        return odeint(self, x0, t)


class NonlinearOscillatorDataset(Dataset):
    def __init__(self, file=None, adjacency_matrix=None, n_samples=1000, n_forecast=5, delta=0.1, single_initial_condition=False):
        if file is not None:
            data = np.load(file)
            data_info = file.replace('.npz', '_info.json')
            with open(data_info, 'r') as f:
                info = json.load(f)
                self.n_samples = info['n_samples']
                self.n_forecast = info['n_forecast']
                self.delta = info['delta']
            self.data = torch.from_numpy(data['data'])
            self.t = torch.from_numpy(data['t'])
            self.adjacency_matrix = torch.from_numpy(data['adjacency_matrix'])
            self.num_nodes = self.adjacency_matrix.shape[0]

        else:
            self.adjacency_matrix = adjacency_matrix
            self.num_nodes = self.adjacency_matrix.shape[0]
            self.n_samples = n_samples
            self.n_forecast = n_forecast
            self.delta = delta
            self.t = torch.linspace(0, self.n_forecast * self.delta, self.n_forecast + 1)
            if single_initial_condition:
                self.data = self.create_data_single_initial_condition()
            else:
                self.data = self.create_data()

    def create_data(self):
        oscillator = NonlinearOscillator(self.adjacency_matrix)
        x_train = 2 * torch.rand(self.n_samples, self.num_nodes, 2) - 1
        data = []
        for x0 in x_train:
            x = oscillator.ode_solve(x0, self.t)
            data.append(x)
        return torch.stack(data)

    def create_data_single_initial_condition(self):
        oscillator = NonlinearOscillator(self.adjacency_matrix)
        t = torch.linspace(0, (self.n_samples + self.n_forecast) * self.delta, (self.n_samples + self.n_forecast))
        x0 = 2 * torch.rand(self.num_nodes, 2) - 1
        x = oscillator.ode_solve(x0, t)
        data = []
        for i in range(len(t) - self.n_forecast):
            data.append(x[i:i + self.n_forecast + 1])
        return torch.stack(data)

    def save_data(self, file):
        np.savez(file, data=self.data.numpy(), t=self.t.numpy(), adjacency_matrix=self.adjacency_matrix.numpy())
        info_dir = {
            'n_samples': self.n_samples,
            'n_forecast': self.n_forecast,
            'delta': self.delta
        }
        with open(file.replace('.npz', '_info.json'), 'w') as f:
            json.dump(info_dir, f)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx, 0, :, :], self.data[idx, 1:, :, :]


if __name__ == '__main__':
    # Generate dataset:
    n_forecast = 5
    delta = 0.1
    n_samples_train = 1000
    n_samples_test = 50

    adjacency_matrix = torch.tensor([[0, 1, 1],
                                     [1, 0, 0],
                                     [0, 1, 0]])

    adjacency_matrix = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

    dataset_train = NonlinearOscillatorDataset(adjacency_matrix=adjacency_matrix, n_samples=n_samples_train, n_forecast=n_forecast, delta=delta, single_initial_condition=True)
    dataset_test = NonlinearOscillatorDataset(adjacency_matrix=adjacency_matrix, n_samples=n_samples_test, n_forecast=n_forecast, delta=delta, single_initial_condition=True)

    dataset_train.save_data('data/train_11node_single_initial_condition.npz')
    dataset_test.save_data('data/test_11node_single_initial_condition.npz')
