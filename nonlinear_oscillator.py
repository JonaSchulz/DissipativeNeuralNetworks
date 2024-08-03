from dataset import NonlinearOscillator, KuramotoOscillator, HarmonicOscillator, RingNetwork
from dissipativity import NodeDynamics, L2Gain, Dissipativity
import matplotlib.pyplot as plt
import torch
import numpy as np


alpha, beta, k = 0.2, 1, 0.1

dynamics = NodeDynamics(alpha=alpha, beta=beta, k=k)
supply_rate = L2Gain()
storage = Dissipativity(dynamics, supply_rate, degree=4)
coefficients, gamma = storage.find_storage_function()
print(gamma)

# Define the adjacency matrix of shape (num_nodes, num_nodes)
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

# adjacency_matrix = torch.tensor([[0, 1, 1],
#                                  [1, 0, 0],
#                                  [0, 1, 0]])

num_nodes = adjacency_matrix.shape[0]

# oscillator = NonlinearOscillator(adjacency_matrix)
# oscillator = HarmonicOscillator(adjacency_matrix, c=1, m=1, k=1)
oscillator = RingNetwork(adjacency_matrix=adjacency_matrix, alpha=alpha, beta=beta, k=k)

t = torch.arange(0, 100, 0.1)
# x0 = torch.tensor([[0.9, 0.],
#                    [0.8, 0.],
#                    [0.7, 0.]])

# x0 = torch.randn(num_nodes, 2)
x0 = 2 * torch.rand(num_nodes, 2) - 1


def u(t, eps=1.0):
    if torch.abs(t.round() - t) < eps and not int(t.item()) % 50:
        return 1.0
    return 0.0

def u2(t):
    return 0.1 * torch.sin(0.1 * t)

x = oscillator.ode_solve(x0, t, u=None)

for i in range(num_nodes):
    plt.plot(t, x[:, i, 1].detach().numpy())

# plt.plot(t, x[:, 0, 1].detach().numpy(), label='v')
plt.legend()
plt.show()


def compute_u(oscillator, x):
    u_values = []
    for t in range(x.shape[0]):
        u_t = -oscillator.beta * torch.sum(oscillator.adjacency_matrix * (x[t, :, 1].reshape(-1, 1) - x[t, :, 1].reshape(1, -1)), dim=1)
        u_values.append(u_t)
    return torch.stack(u_values)


u = compute_u(oscillator, x)
V_x = storage.evaluate_dissipativity(x.detach().numpy(), u.detach().numpy())
# supply = supply_rate.evaluate(u, x[:, :, 1].detach().numpy())

plt.plot(t, V_x[:, 0])
# plt.plot(t, supply[:, 0])
plt.show()

# # Plot the trajectories of each node in three subplots
# fig, axs = plt.subplots(3, 1, figsize=(10, 10))
# for i in range(num_nodes):
#     axs[i].plot(x[:, i, 0].detach().numpy(), x[:, i, 1].detach().numpy(), label=f'Node {i + 1}')
#     axs[i].set_xlabel('Position')
#     axs[i].set_ylabel('Velocity')
#     axs[i].legend()
#
# plt.tight_layout()
# plt.show()
