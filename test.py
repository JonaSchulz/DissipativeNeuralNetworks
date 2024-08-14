from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import mlflow
import os
from urllib.parse import urlparse
import ast

from dissnn.model import NetworkODEModel, DissipativityLoss, SparsityLoss
from dissnn.dataset import NonlinearOscillatorDataset, NonlinearOscillator2, NonlinearPendulum
from dissnn.dissipativity import Dissipativity, NonlinearOscillator2NodeDynamics, L2Gain, DissipativityPendulum

run_id = '2359f9703054477a88247bd0dff216c2'
t_max = 40.0  # Maximum time for simulation
t_step = 0.01   # Time step for simulation
axis_label_fontsize = 16
title_fontsize = 18
legend_fontsize = 14
axis_tick_fontsize = 12
pendulum = True
plt.style.use('ggplot')

with mlflow.start_run(run_id=run_id) as run:
    run_params = run.data.to_dictionary()['params']

    model_file = os.path.join(urlparse(run.info.artifact_uri).path, 'best_model', 'data', 'model.pth')
    test_data_file = run_params['dataset_test']
    use_gt_adjacency_matrix = bool(run_params['use_gt_adjacency_matrix'])
    batch_size = 1
    device = 'cuda'
    sparsity_weight = float(run_params['sparsity_weight'])
    dissipativity_weight = float(run_params['dissipativity_weight'])

    NodeDynamics = NonlinearOscillator2NodeDynamics

    # Create test data loaders:
    dataset_test = NonlinearOscillatorDataset(file=test_data_file)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Define the model:
    num_nodes = dataset_test.num_nodes
    hidden_dim_node = int(run_params['hidden_dim_node'])
    num_hidden_layers_node = int(run_params['num_hidden_layers_node'])
    hidden_dim_coupling = int(run_params['hidden_dim_coupling'])
    num_hidden_layers_coupling = int(run_params['num_hidden_layers_coupling'])
    adjacency_matrix = dataset_test.adjacency_matrix.to(float).to(device) if use_gt_adjacency_matrix else None

    # Model:
    # model = NetworkODEModel(num_nodes=num_nodes,
    #                         input_dim=2,
    #                         output_dim_nn=1,
    #                         hidden_dim_node=hidden_dim_node,
    #                         num_hidden_layers_node=num_hidden_layers_node,
    #                         hidden_dim_coupling=hidden_dim_coupling,
    #                         num_hidden_layers_coupling=num_hidden_layers_coupling,
    #                         adjacency_matrix=adjacency_matrix).to(device)

    # model.load(model_file)
    model = torch.load(model_file)
    model.eval()

    # Ground Truth Dynamical System:
    if pendulum:
        oscillator = NonlinearPendulum(dataset_test.adjacency_matrix.to(device), device=device, **dataset_test.info)
    else:
        oscillator = NonlinearOscillator2(dataset_test.adjacency_matrix.to(device), device=device, **dataset_test.info)

    # Dissipativity:
    if pendulum:
        dissipativity = DissipativityPendulum(**dataset_test.info)
    else:
        dynamics = NodeDynamics(**dataset_test.info)
        supply_rate = L2Gain()
        dissipativity = Dissipativity(dynamics, supply_rate, degree=int(run_params['storage_function_degree']))
        dissipativity.coefficients = ast.literal_eval(run_params['storage_coefficients'])
        dissipativity.coefficient_values = ast.literal_eval(run_params['storage_coefficient_values'])
        if isinstance(supply_rate, L2Gain):
            dissipativity.supply_rate.gamma_value = float(run_params['gamma_value'])
    # dissipativity.find_storage_function()

    # Loss:
    criterion = SparsityLoss(model, alpha=sparsity_weight).to(device)
    criterion_dissipativity = DissipativityLoss(dissipativity, dataset_test.adjacency_matrix, device=device).to(device)

    with torch.no_grad():
        # Simulate a trajectory and test the model on it:
        x0, _ = next(iter(dataloader_test))
        x0 = x0.to(device)
        t = torch.arange(0, t_max, t_step).to(device)
        x_gt = oscillator.ode_solve(x0.squeeze(), t).unsqueeze(0)
        x_pred = model(x0, t)

        # Plot ground truth vs prediction trajectory for node 1:
        plt.plot(x_pred[0, :, 0, 0].cpu().detach().numpy(), x_pred[0, :, 0, 1].cpu().detach().numpy(),
                 label='Prediction', color='blue')
        plt.plot(x_gt[0, :, 0, 0].cpu().detach().numpy(), x_gt[0, :, 0, 1].cpu().detach().numpy(),
                 label='Ground Truth')
        plt.xlabel('x1', fontsize=axis_label_fontsize)
        plt.ylabel('x2', fontsize=axis_label_fontsize)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=5,
                   fontsize=legend_fontsize)
        plt.subplots_adjust(bottom=0.25)
        plt.title('Node 1 Trajectory', fontsize=title_fontsize)
        plt.tick_params(axis='x', labelsize=axis_tick_fontsize)
        plt.tick_params(axis='y', labelsize=axis_tick_fontsize)
        plt.show()

        # Plot dissipativity violations on trajectories for node 1:
        dissipativity_loss = criterion_dissipativity(x_pred, model, aggregator=None, relu=False)[:, 0]
        violation_indices = dissipativity_loss < 0.0
        plt.scatter(x_pred[0, violation_indices, 0, 0].cpu().detach().numpy(),
                    x_pred[0, violation_indices, 0, 1].cpu().detach().numpy(), s=20, color='red',
                    label='Dissipativity Violation')
        plt.plot(x_pred[0, :, 0, 0].cpu().detach().numpy(), x_pred[0, :, 0, 1].cpu().detach().numpy(),
                 label='Prediction', color='blue')
        plt.xlabel('x1', fontsize=axis_label_fontsize)
        plt.ylabel('x2', fontsize=axis_label_fontsize)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=5,
                   fontsize=legend_fontsize)
        plt.subplots_adjust(bottom=0.25)
        plt.title('Node 1 Dissipativity Violations', fontsize=title_fontsize)
        plt.tick_params(axis='x', labelsize=axis_tick_fontsize)
        plt.tick_params(axis='y', labelsize=axis_tick_fontsize)
        plt.show()

        # Plot dissipativity loss over time for node 1:
        plt.axhline(y=0, color='red', linestyle='--')
        plt.plot(t.cpu().numpy(), dissipativity_loss.cpu().numpy(), label='Dissipativity Loss', color='blue')
        plt.xlabel('Time', fontsize=axis_label_fontsize)
        plt.ylabel('$s(u,y)-\\dot{V}(x)$', fontsize=axis_label_fontsize)
        plt.title('Node 1 Dissipativity Loss', fontsize=title_fontsize)
        plt.tick_params(axis='x', labelsize=axis_tick_fontsize)
        plt.tick_params(axis='y', labelsize=axis_tick_fontsize)
        plt.show()

        # Plot storage function over time for node 1:
        storage = dissipativity.evaluate_storage(x_pred.squeeze())[:, 0]
        plt.plot(t.cpu().numpy(), storage.cpu().numpy(), label='Storage Function', color='blue')
        plt.xlabel('Time', fontsize=axis_label_fontsize)
        plt.ylabel('V(x)', fontsize=axis_label_fontsize)
        plt.title('Node 1 Storage Function', fontsize=title_fontsize)
        plt.tick_params(axis='x', labelsize=axis_tick_fontsize)
        plt.tick_params(axis='y', labelsize=axis_tick_fontsize)
        plt.show()

    exit()
    # Plot the ground-truth and predicted trajectories of each node:
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    for i in range(3):
        axs[i].plot(x_gt[:, i, 0].cpu().detach().numpy(), x_gt[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} GT')
        axs[i].plot(x_pred[:, i, 0].cpu().detach().numpy(), x_pred[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} Pred')
        axs[i].set_xlabel('Position')
        axs[i].set_ylabel('Velocity')
        axs[i].legend()

    plt.tight_layout()
    plt.show()

    # Plot the ground-truth and predicted state evolution over time of each node for a sample from the test dataset:
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    for i in range(3):
        axs[i].plot(t.cpu().numpy(), x_gt[:, i, 0].cpu().detach().numpy(), label=f'Node {i + 1} GT Position')
        axs[i].plot(t.cpu().numpy(), x_pred[:, i, 0].cpu().detach().numpy(), label=f'Node {i + 1} Pred Position')
        axs[i].plot(t.cpu().numpy(), x_gt[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} GT Velocity')
        axs[i].plot(t.cpu().numpy(), x_pred[:, i, 1].cpu().detach().numpy(), label=f'Node {i + 1} Pred Velocity')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Position/Velocity')
        axs[i].legend()

    plt.tight_layout()
    plt.show()
