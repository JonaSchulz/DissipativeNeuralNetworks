import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint


class NodeNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(NodeNetwork, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
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
        self.hidden_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
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

            # TODO: parallelize by feeding all pairs of x_i, x_j to the coupling network at the same time
            for j in range(self.num_nodes):
                # Use the adjacency matrix element
                A_ij = adjacency_matrix[i, j]
                coupling_output = self.coupling_network(x[:, i, :], x[:, j, :])
                coupling_sum += A_ij * coupling_output

            output[:, i, 1] = node_output[:, 0] + coupling_sum[:, 0]    # v_dot = f(x_i) + sum(A_ij * g(x_i, x_j))

        output[:, :, 0] = x[:, :, 1]    # x_dot = v

        return output

    def evaluate_parallel(self, t, x):
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

        # Apply the node_network to each node:
        node_output = self.node_network(x.reshape(-1, node_dim)).reshape(batch_size, self.num_nodes, self.output_dim_nn)

        # Apply the coupling network to all pairs of nodes:
        x1 = x.unsqueeze(1).repeat_interleave(self.num_nodes, dim=1)
        x2 = x.unsqueeze(2).repeat_interleave(self.num_nodes, dim=2)
        pairwise_combinations = torch.cat((x1.unsqueeze(3), x2.unsqueeze(3)), dim=3)
        pairwise_combinations = pairwise_combinations.reshape(batch_size, self.num_nodes ** 2, 2 * node_dim)
        coupling_output = self.coupling_network(pairwise_combinations)
        coupling_output = coupling_output.reshape(batch_size, self.num_nodes, self.num_nodes, self.output_dim_nn)

        output[:, :, 1] = node_output[:, :, 0] + torch.sum(adjacency_matrix.unsqueeze(0).unsqueeze(3) * coupling_output, dim=2)[:, :, 0]    # v_dot = f(x_i) + sum(A_ij * g(x_i, x_j))
        output[:, :, 0] = x[:, :, 1]    # x_dot = v

        return output

    def get_adjacency_matrix(self):
        A = F.sigmoid(self.adjacency_matrix_parameter - torch.eye(self.num_nodes).to(self.adjacency_matrix_parameter.device) / self.eps)
        if not self.training:
            A = torch.round(A)
        return A

    def load(self, file, adjacency_matrix=None):
        state_dict = torch.load(file)
        if adjacency_matrix is not None:
            state_dict['adjacency_matrix_parameter'] = adjacency_matrix.to(self.adjacency_matrix_parameter.device)
        self.load_state_dict(state_dict)

    def forward(self, x0, t):
        x_pred = odeint(self.evaluate_parallel, x0, t)
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


class DissipativityLoss(nn.Module):
    def __init__(self, dissipativity, adjacency_matrix, device='cpu'):
        super(DissipativityLoss, self).__init__()
        self.dissipativity = dissipativity
        self.adjacency_matrix = adjacency_matrix.to(device)
        self.device = device

    def compute_u(self, x):
        """
        Compute the control input u for the dissipativity evaluation (u_i = sum_j A_ij * (x_j[2] - x_i[2]))
        """
        u_values = []
        x = x.reshape(-1, x.shape[2], x.shape[3])
        for t in range(x.shape[0]):
            u_t = torch.sum(self.adjacency_matrix * (x[t, :, 1].reshape(-1, 1) - x[t, :, 1].reshape(1, -1)), dim=1)
            u_values.append(u_t)
        return torch.stack(u_values).unsqueeze(-1).to(self.device)

    def forward(self, x_pred, model):
        u = self.compute_u(x_pred)
        x_pred = x_pred.reshape(-1, x_pred.shape[2], x_pred.shape[3])
        x_dot = model.evaluate_parallel(None, x_pred)
        dissipativity = self.dissipativity.evaluate_dissipativity(x_pred, u, x_dot)
        return F.relu(-dissipativity).mean()


if __name__ == '__main__':
    from dissipativity import Dissipativity

    dissipativity = Dissipativity()
    loss = DissipativityLoss()