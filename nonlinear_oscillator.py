from model import NonlinearOscillator
import matplotlib.pyplot as plt
import torch


# Define the adjacency matrix of shape (num_nodes, num_nodes)
num_nodes = 3
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
t = torch.arange(0, 50, 0.1)
x0 = torch.tensor([[0.9, 0.],
                   [0.8, 0.],
                   [0.7, 0.]])
# x0 = torch.tensor([[1.0, 0.] for i in range(11)])

# Solve the ODE system for each node
x = oscillator.ode_solve(x0, t)

plt.plot(t, x[:, 0, 0].detach().numpy(), label='x')
plt.plot(t, x[:, 0, 1].detach().numpy(), label='v')
plt.legend()
plt.show()

# Plot the trajectories of each node in three subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
for i in range(num_nodes):
    axs[i].plot(x[:, i, 0].detach().numpy(), x[:, i, 1].detach().numpy(), label=f'Node {i + 1}')
    axs[i].set_xlabel('Position')
    axs[i].set_ylabel('Velocity')
    axs[i].legend()

plt.tight_layout()
plt.show()

