import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint
from torch.utils.data import Dataset
import json


class NonlinearOscillator:
    def __init__(self, adjacency_matrix, a1=0.2, a2=11.0, a3=11.0, a4=1.0, device='cpu'):
        self.adjacency_matrix = adjacency_matrix.to(device)
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.node_dim = 2
        self.device = device

    def __call__(self, t, x):
        out = torch.empty_like(x).to(self.device)
        out[:, 0] = x[:, 1]
        out[:, 1] = -x[:, 0] - self.a1 * x[:, 1] * (self.a2 * x[:, 0] ** 4 - self.a3 * x[:, 0] + self.a4)
        out[:, 1] += torch.sum(self.adjacency_matrix * (x[:, 1].reshape(-1, 1) - x[:, 1].reshape(1, -1)), dim=1)
        return out

    def ode_solve(self, x0, t):
        return odeint(self, x0, t)

    def info_dir(self):
        return {
            'a1': self.a1,
            'a2': self.a2,
            'a3': self.a3,
            'a4': self.a4,
            'node_dim': self.node_dim
        }


class KuramotoOscillator:
    def __init__(self, adjacency_matrix, omega=1, device='cpu', k=1.0):
        self.adjacency_matrix = adjacency_matrix.to(device)
        self.device = device
        self.omega = omega
        self.k = k
        self.node_dim = 1

    def __call__(self, t, x):
        out = torch.empty_like(x).to(self.device)
        out[:, 0] = self.omega + self.k * torch.mean(self.adjacency_matrix * torch.sin(x[:, 0].reshape(1, -1) - x[:, 0].reshape(-1, 1)), dim=1)
        return out

    def ode_solve(self, x0, t):
        return odeint(self, x0, t)

    def info_dir(self):
        return {
            'omega': self.omega,
            'k': self.k,
            'node_dim': self.node_dim
        }


class HarmonicOscillator:
    def __init__(self, adjacency_matrix, c=1, m=1, k=1):
        self.adjacency_matrix = adjacency_matrix
        self.c = c
        self.m = m
        self.k = k
        self.node_dim = 2

    def __call__(self, t, x):
        out = torch.empty_like(x)
        out[:, 0] = x[:, 1]
        out[:, 1] = -self.c / self.m * x[:, 1] - self.k / self.m * x[:, 0]
        out[:, 1] -= torch.sum(self.adjacency_matrix * (x[:, 1].reshape(-1, 1) - x[:, 1].reshape(1, -1)), dim=1)
        return out

    def ode_solve(self, x0, t):
        return odeint(self, x0, t)

    def info_dir(self):
        return {
            'c': self.c,
            'm': self.m,
            'k': self.k,
            'node_dim': self.node_dim
        }


class NonlinearOscillator2:
    def __init__(self, adjacency_matrix, alpha=1.0, beta=1.0, k=1.0, **kwargs):
        self.adjacency_matrix = adjacency_matrix
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.u = None
        self.node_dim = 2

    def __call__(self, t, x):
        out = torch.empty_like(x)
        out[:, 0] = x[:, 1]
        out[:, 1] = -self.alpha * x[:, 0] ** 3 - self.k * x[:, 1]   # + self.beta * (x[:, 1].roll(1, 0) - x[:, 1])
        u = -self.beta * torch.sum(self.adjacency_matrix * (x[:, 1].reshape(-1, 1) - x[:, 1].reshape(1, -1)), dim=1)
        out[:, 1] += u

        if self.u is not None:
            out[0, 1] += self.u(t)
        return out

    def evaluate_parallel(self, t, x):
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            out[i, :, 0] = x[i, :, 1]
            out[i, :, 1] = -self.alpha * x[i, :, 0] ** 3 - self.k * x[i, :, 1]   # + self.beta * (x[:, 1].roll(1, 0) - x[:, 1])
            u = -self.beta * torch.sum(self.adjacency_matrix * (x[i, :, 1].reshape(-1, 1) - x[i, :, 1].reshape(1, -1)), dim=1)
            out[i, :, 1] += u

            if self.u is not None:
                out[i, 0, 1] += self.u(t)
        return out

    def ode_solve(self, x0, t, u=None):
        self.u = u
        return odeint(self, x0, t)

    def info_dir(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'k': self.k,
            'node_dim': self.node_dim
        }

class NonlinearOscillator3:
    def __init__(self, adjacency_matrix, alpha=1.0, beta=1.0, k=1.0, **kwargs):
        self.adjacency_matrix = adjacency_matrix
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.u = None
        self.node_dim = 2

    def __call__(self, t, x):
        out = torch.empty_like(x)
        out[:, 0] = x[:, 1]
        out[:, 1] = -self.alpha * x[:, 0] ** 3 - self.k * x[:, 1]   # + self.beta * (x[:, 1].roll(1, 0) - x[:, 1])
        u = -self.beta * torch.sum(self.adjacency_matrix * (x[:, 1].reshape(-1, 1)**2 - x[:, 1].reshape(1, -1)**2), dim=1) # xj^2 - xi^2
        out[:, 1] += u

        if self.u is not None:
            out[0, 1] += self.u(t)
        return out

    def evaluate_parallel(self, t, x):
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            out[i, :, 0] = x[i, :, 1]
            out[i, :, 1] = -self.alpha * x[i, :, 0] ** 3 - self.k * x[i, :, 1]   # + self.beta * (x[:, 1].roll(1, 0) - x[:, 1])
            u = -self.beta * torch.sum(self.adjacency_matrix * (x[i, :, 1].reshape(-1, 1)**2 - x[i, :, 1].reshape(1, -1)**2), dim=1) # xj^2 - xi^2
            out[i, :, 1] += u

            if self.u is not None:
                out[i, 0, 1] += self.u(t)
        return out

    def ode_solve(self, x0, t, u=None):
        self.u = u
        return odeint(self, x0, t)

    def info_dir(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'k': self.k,
            'node_dim': self.node_dim
        }



class LotkaVolterra:
    def __init__(self, adjacency_matrix, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        self.adjacency_matrix = adjacency_matrix
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.node_dim = 2

    def __call__(self, t, x):
        out = torch.empty_like(x)
        out[:, 0] = self.alpha * x[:, 0] - self.beta * x[:, 0] * x[:, 1]
        out[:, 1] = -self.gamma * x[:, 1] + self.delta * x[:, 0] * x[:, 1]
        out[:, 0] -= torch.sum(self.adjacency_matrix * (x[:, 0].reshape(-1, 1) - x[:, 0].reshape(1, -1)), dim=1)
        out[:, 1] -= torch.sum(self.adjacency_matrix * (x[:, 1].reshape(-1, 1) - x[:, 1].reshape(1, -1)), dim=1)
        return out

    def ode_solve(self, x0, t):
        return odeint(self, x0, t)

    def info_dir(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
            'node_dim': self.node_dim
        }


class NonlinearOscillatorDataset(Dataset):
    def __init__(self, file=None, adjacency_matrix=None, network_model=None, n_samples=1000, n_forecast=5, delta=0.1,
                 single_initial_condition=False, num_trajectories=10):
        if file is not None:
            data = np.load(file)
            data_info = file.replace('.npz', '_info.json')
            with open(data_info, 'r') as f:
                self.info = json.load(f)
                self.n_samples = self.info['n_samples']
                self.n_forecast = self.info['n_forecast']
                self.delta = self.info['delta']
            self.data = torch.from_numpy(data['data'])
            self.t = torch.from_numpy(data['t'])
            self.adjacency_matrix = torch.from_numpy(data['adjacency_matrix'])
            self.num_nodes = self.adjacency_matrix.shape[0]

        else:
            self.adjacency_matrix = adjacency_matrix
            self.network = network_model
            self.num_nodes = self.adjacency_matrix.shape[0]
            self.n_samples = n_samples
            self.n_forecast = n_forecast
            self.delta = delta
            self.t = torch.linspace(0, self.n_forecast * self.delta, self.n_forecast + 1)
            if single_initial_condition:
                self.data = self.create_data_single_initial_condition()
            else:
                self.data = self.create_data(num_trajectories)

    def create_data_single_initial_condition(self):
        t = torch.linspace(0, (self.n_samples + self.n_forecast) * self.delta, (self.n_samples + self.n_forecast))
        x0 = 2 * torch.rand(self.num_nodes, self.network.node_dim) - 1
        x = self.network.ode_solve(x0, t)
        data = []
        for i in range(len(t) - self.n_forecast):
            data.append(x[i:i + self.n_forecast + 1])
        return torch.stack(data)

    def create_data(self, num_trajectories=10):
        samples_per_trajectory = self.n_samples // num_trajectories
        t = torch.linspace(0, (samples_per_trajectory + self.n_forecast) * self.delta, (samples_per_trajectory + self.n_forecast))
        data = []
        for trajectory in range(num_trajectories):
            x0 = 2 * torch.rand(self.num_nodes, self.network.node_dim) - 1
            x = self.network.ode_solve(x0, t)
            for i in range(len(t) - self.n_forecast):
                data.append(x[i:i + self.n_forecast + 1])
        data = torch.stack(data)
        self.n_samples = len(data)
        return data

    def save_data(self, file):
        np.savez(file, data=self.data.numpy(), t=self.t.numpy(), adjacency_matrix=self.adjacency_matrix.numpy())
        info_dir = {
            'n_samples': self.n_samples,
            'n_forecast': self.n_forecast,
            'delta': self.delta
        }
        info_dir.update(self.network.info_dir())
        self.info = info_dir
        with open(file.replace('.npz', '_info.json'), 'w') as f:
            json.dump(info_dir, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, 0, :, :], self.data[idx, 1:, :, :]


if __name__ == '__main__':
    # Generate dataset:
    n_forecast = 5
    delta = 0.1
    n_samples_train = 5000
    n_samples_test = 100

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

    # adjacency_matrix = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                                  [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    #                                  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #                                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                                  [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    #                                  [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
    #                                  [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    #                                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                                  [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #                                  [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    #                                  [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]])

    model = NonlinearOscillator3(adjacency_matrix=adjacency_matrix, alpha=0.1, beta=0.01, k=0.01)
    # model = NonlinearOscillator(adjacency_matrix=adjacency_matrix)

    dataset_train = NonlinearOscillatorDataset(adjacency_matrix=adjacency_matrix, network_model=model, n_samples=n_samples_train, n_forecast=n_forecast, delta=delta, single_initial_condition=False, num_trajectories=50)
    dataset_test = NonlinearOscillatorDataset(adjacency_matrix=adjacency_matrix, network_model=model, n_samples=n_samples_test, n_forecast=n_forecast, delta=delta, single_initial_condition=False, num_trajectories=10)

    dataset_train.save_data('../data/oscillator3_11node/train.npz')
    dataset_test.save_data('../data/oscillator3_11node/test.npz')
