from dataset import NonlinearOscillator, KuramotoOscillator
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

oscillator = KuramotoOscillator(adjacency_matrix, omega=1)

t = torch.arange(0, 10, 0.1)
x0 = torch.tensor([[0.9],
                   [0.8],
                   [0.7]])

# x0 = torch.tensor([[1.0, 0.] for i in range(11)])

# Solve the ODE system for each node
x = oscillator.ode_solve(x0, t)

x = (x + 2 * torch.pi) % (2 * torch.pi) - torch.pi
for i in range(3):
    plt.plot(t, x[:, i, 0].detach().numpy(), label=f'$\\theta_{i}$')
plt.legend()
plt.show()
