from dissnn.dataset import LotkaVolterra, NonlinearOscillator2, NonlinearPendulum
import matplotlib.pyplot as plt
import torch

alpha, beta, gamma, delta = 1.0, 1.0, 1.0, 1.0

#dynamics = NodeDynamics(alpha=alpha, beta=beta, k=k)
#supply_rate = L2Gain()
#storage = Dissipativity(dynamics, supply_rate, degree=4)
#coefficients, gamma = storage.find_storage_function()
#print(gamma)

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

adjacency_matrix = torch.tensor([[0, 1, 1],
                                 [1, 0, 0],
                                 [0, 1, 0]])

# adjacency_matrix = torch.tensor([[0, 0, 0],
#                                  [0, 0, 0],
#                                  [0, 0, 0]])


num_nodes = adjacency_matrix.shape[0]

# oscillator = NonlinearOscillator(adjacency_matrix)
# oscillator = HarmonicOscillator(adjacency_matrix, c=1, m=1, k=1)
# oscillator = NonlinearOscillator2(adjacency_matrix=adjacency_matrix, alpha=0.1, beta=0.01, k=0.01)
oscillator = NonlinearPendulum(adjacency_matrix=adjacency_matrix, d=1.0, m=1.0, k=1.0)

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

x = oscillator.ode_solve(x0, t)

axis_label_fontsize = 16
title_fontsize = 18
legend_fontsize = 14
axis_tick_fontsize = 12
plt.style.use('ggplot')

for i in range(1):
    plt.plot(t, x[:, i, 0].detach().numpy(), label="$\\theta$")
    plt.plot(t, x[:, i, 1].detach().numpy(), label="$\\omega$")

plt.xlabel('Time', fontsize=axis_label_fontsize)
# plt.ylabel('x0', fontsize=axis_label_fontsize)
# plt.title('Node 1 Storage Function', fontsize=title_fontsize)
plt.tick_params(axis='x', labelsize=axis_tick_fontsize)
plt.tick_params(axis='y', labelsize=axis_tick_fontsize)
plt.tight_layout()

# plt.plot(t, x[:, 0, 1].detach().numpy(), label='v')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=5,
                   fontsize=legend_fontsize)
plt.subplots_adjust(bottom=0.25)
plt.savefig("pendulum.svg")


# # Plot the trajectories of each node in three subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
for i in range(3):
     axs[i].plot(x[:, i, 0].detach().numpy(), x[:, i, 1].detach().numpy(), label=f'Node {i + 1}')
     axs[i].set_xlabel('Position')
     axs[i].set_ylabel('Velocity')
     axs[i].legend()

plt.tight_layout()
plt.show()
