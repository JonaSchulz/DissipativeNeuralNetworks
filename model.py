import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint


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
        # TODO: possible error
        out[:, 1] += torch.sum(self.adjacency_matrix * (x[:, 1].reshape(-1, 1) - x[:, 1].reshape(1, -1)), dim=1)
        return out

    def ode_solve(self, x0, t):
        return odeint(self, x0, t)


class NodeNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(NodeNetwork, self).__init__()
        self.hidden_layers = [nn.Linear(input_dim, hidden_dim)]
        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


class CouplingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(CouplingNetwork, self).__init__()
        self.hidden_layers = [nn.Linear(input_dim, hidden_dim)]
        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


class NetworkODEModel(nn.Module):
    def __init__(self,
                 num_nodes,
                 input_dim,
                 output_dim,
                 hidden_dim_node,
                 num_hidden_layers_node,
                 hidden_dim_coupling,
                 num_hidden_layers_coupling,
                 eps=1e-5):
        super(NetworkODEModel, self).__init__()
        self.node_network = NodeNetwork(input_dim, output_dim, hidden_dim_node, num_hidden_layers_node)
        self.coupling_network = CouplingNetwork(2 * input_dim, input_dim, hidden_dim_coupling, num_hidden_layers_coupling)
        self.adjacency_matrix_parameter = nn.Parameter(torch.zeros((num_nodes, num_nodes)))
        self.num_nodes = num_nodes
        self.eps = eps

    def evaluate(self, x):
        """

        :param x: shape (batch_size, num_nodes, node_dim)
        :return: x_dot with shape (batch_size, num_nodes, node_dim)
        """

        batch_size = x.size(0)
        node_dim = x.size(-1)

        # Initialize the output tensor
        output = torch.zeros_like(x)

        # Generate differentiable adjacency matrix from adjacency_matrix_parameter
        adjacency_matrix = F.sigmoid(self.adjacency_matrix_parameter - torch.eye(self.num_nodes) / self.eps)

        # Apply the node_network to each node
        for i in range(self.num_nodes):
            node_output = self.node_network(x[:, i, :])
            coupling_sum = torch.zeros((batch_size, node_dim), device=x.device)

            for j in range(self.num_nodes):
                # Use the adjacency matrix element
                A_ij = adjacency_matrix[i, j]
                coupling_output = self.coupling_network(x[:, i, :], x[:, j, :])
                coupling_sum += A_ij * coupling_output

            output[:, i, 1] = node_output + coupling_sum    # v_dot = node + coupling

        output[:, :, 0] = x[:, :, 1]    # x_dot = v

        return output

    def forward(self, x0, t):
        x_pred = odeint(self.evaluate, x0, t)
        return x_pred
