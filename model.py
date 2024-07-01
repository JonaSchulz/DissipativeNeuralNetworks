import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint


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
                 output_dim_nn,
                 hidden_dim_node,
                 num_hidden_layers_node,
                 hidden_dim_coupling,
                 num_hidden_layers_coupling,
                 eps=1e-5):
        super(NetworkODEModel, self).__init__()
        self.node_network = NodeNetwork(input_dim, output_dim_nn, hidden_dim_node, num_hidden_layers_node)
        self.coupling_network = CouplingNetwork(2 * input_dim, output_dim_nn, hidden_dim_coupling, num_hidden_layers_coupling)
        self.adjacency_matrix_parameter = nn.Parameter(torch.zeros((num_nodes, num_nodes)))
        self.num_nodes = num_nodes
        self.output_dim_nn = output_dim_nn
        self.eps = eps

    def evaluate(self, t, x):
        """

        :param x: shape (batch_size, num_nodes, node_dim)
        :return: x_dot with shape (batch_size, num_nodes, node_dim)
        """

        batch_size = x.size(0)
        node_dim = x.size(-1)

        # Initialize the output tensor
        output = torch.zeros_like(x)

        # Generate differentiable adjacency matrix from adjacency_matrix_parameter
        adjacency_matrix = self.get_adjacency_matrix()

        # Apply the node_network to each node
        for i in range(self.num_nodes):
            node_output = self.node_network(x[:, i, :])
            coupling_sum = torch.zeros((batch_size, self.output_dim_nn), device=x.device)

            for j in range(self.num_nodes):
                # Use the adjacency matrix element
                A_ij = adjacency_matrix[i, j]
                coupling_output = self.coupling_network(x[:, i, :], x[:, j, :])
                coupling_sum += A_ij * coupling_output

            output[:, i, 1] = node_output[:, 0] + coupling_sum[:, 0]    # v_dot = f(x_i) + sum(A_ij * g(x_i, x_j))

        output[:, :, 0] = x[:, :, 1]    # x_dot = v

        return output

    def get_adjacency_matrix(self):
        return F.sigmoid(self.adjacency_matrix_parameter - torch.eye(self.num_nodes) / self.eps)

    def forward(self, x0, t):
        x_pred = odeint(self.evaluate, x0, t)
        x_pred = x_pred.permute(1, 0, 2, 3)
        return x_pred


class SparsityLoss(nn.Module):
    def __init__(self, model, alpha=0.0):
        super(SparsityLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, x_pred, x_gt):
        adjacency_matrix = self.model.get_adjacency_matrix()
        return self.mse(x_pred, x_gt) + self.alpha * torch.norm(adjacency_matrix, p=1)
