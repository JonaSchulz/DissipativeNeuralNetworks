from dataset import NonlinearOscillator, KuramotoOscillator, HarmonicOscillator, RingNetwork
import matplotlib.pyplot as plt
import torch


# Define the adjacency matrix of shape (num_nodes, num_nodes)
num_nodes = 20
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

adjacency_matrix = torch.tensor([[0, 1, 1],
                                 [1, 0, 0],
                                 [0, 1, 0]])

oscillator = NonlinearOscillator(adjacency_matrix)
oscillator = HarmonicOscillator(adjacency_matrix, c=1, m=1, k=1)
oscillator = RingNetwork(num_nodes=num_nodes, alpha=0.01, beta=1)

t = torch.arange(0, 300, 0.1)
x0 = torch.tensor([[0.9, 0.],
                   [0.8, 0.],
                   [0.7, 0.]])

x0 = torch.randn(num_nodes, 1)
x0 = 2 * torch.rand(num_nodes, 1) - 1

u = torch.sin(t)

# x0 = torch.tensor([[1.0, 0.] for i in range(11)])

# Solve the ODE system for each node
print(x0)

def u(t, eps=1.0):
    if torch.abs(t.round() - t) < eps and not int(t.item()) % 50:
        return 1.0
    return 0.0

x = oscillator.ode_solve(x0, t, u=u)

for i in range(num_nodes):
    plt.plot(t, x[:, i, 0].detach().numpy())

# plt.plot(t, x[:, 0, 1].detach().numpy(), label='v')
plt.legend()
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

# Define a storage function being the squared states of all nodes in the RingNetwork for every time step.
# The storage function is a scalar function of time, i.e., it is a tensor of shape (num_time_steps, num_nodes).
storage_function = x[:, :, 0] ** 2

# Define a supply rate function
supply_rate = x[:, :, 0].roll(1, 1) ** 2 - x[:, :, 0] ** 2

# Plot the storage function and the integral of the supply rate from 0 to t for the first node
plt.plot(t, storage_function[:, 0].detach().numpy(), label='Storage Function')
plt.plot(t, torch.cumsum(supply_rate[:, 0], dim=0).detach().numpy(), label='Integral of Supply Rate')
plt.legend()
plt.show()