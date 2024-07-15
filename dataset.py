import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint
from torch.utils.data import Dataset
import json


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


class NonlinearOscillatorDataset(Dataset):
    def __init__(self, file=None, adjacency_matrix=None, n_samples=1000, n_forecast=5, delta=0.1):
        if file is not None:
            data = np.load(file)
            self.data = torch.from_numpy(data['data'])
            self.t = torch.from_numpy(data['t'])
            self.n_samples = self.data.shape[0]
            self.adjacency_matrix = torch.from_numpy(data['adjacency_matrix'])

        else:
            self.adjacency_matrix = adjacency_matrix
            self.n_samples = n_samples
            self.n_forecast = n_forecast
            self.delta = delta
            self.t = torch.linspace(0, self.n_forecast * self.delta, self.n_forecast + 1)
            self.data = self.create_data()

        self.num_nodes = self.adjacency_matrix.shape[0]

    def create_data(self):
        num_nodes = self.adjacency_matrix.shape[0]
        oscillator = NonlinearOscillator(self.adjacency_matrix)
        x_train = 2 * torch.rand(self.n_samples, num_nodes, 2) - 1
        data = []
        for x0 in x_train:
            x = oscillator.ode_solve(x0, self.t)
            data.append(x)
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

    dataset_train = NonlinearOscillatorDataset(adjacency_matrix=adjacency_matrix, n_samples=n_samples_train, n_forecast=n_forecast, delta=delta)
    dataset_test = NonlinearOscillatorDataset(adjacency_matrix=adjacency_matrix, n_samples=n_samples_test, n_forecast=n_forecast, delta=delta)

    dataset_train.save_data('data/train_11node.npz')
    dataset_test.save_data('data/test_11node.npz')
