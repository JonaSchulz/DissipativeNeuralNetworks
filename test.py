from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import mlflow
import os
from urllib.parse import urlparse
import ast
import tempfile

from dissnn.model import NetworkODEModel, DissipativityLoss, SparsityLoss
from dissnn.dataset import NonlinearOscillatorDataset, NonlinearOscillator2, NonlinearPendulum
from dissnn.dissipativity import Dissipativity, NonlinearOscillator2NodeDynamics, L2Gain, DissipativityPendulum

run_id = '09d61d9fdda64cd3bdb434c8dbbccfab'
test_data_file = 'data/oscillator2_11node_3/val.npz'
test_data_file = None
do_eval = False
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
    if test_data_file is None:
        test_data_file = run_params['dataset_test']
    use_gt_adjacency_matrix = bool(run_params['use_gt_adjacency_matrix'])
    batch_size = 128
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
        print(dissipativity.polynomial)
        print(dissipativity.coefficients)
        print(dissipativity.coefficient_values)
    # dissipativity.find_storage_function()

    # Loss:
    criterion = SparsityLoss(model, alpha=sparsity_weight).to(device)
    criterion_dissipativity = DissipativityLoss(dissipativity, dataset_test.adjacency_matrix, device=device).to(device)

    with torch.no_grad():
        # Evaluate on test set:
        if do_eval:
            model.eval()
            loss_mse = 0.0
            loss_dissipativity = 0.0
            percentage = 0.0

            for x0, x_gt in dataloader_test:
                x0 = x0.to(device)
                x_gt = x_gt.to(device)
                x_pred = model(x0, dataset_test.t.to(device))
                loss_mse += criterion(x_pred[:, 1:, :, :], x_gt)
                loss_dissipativity += criterion_dissipativity(x_pred, model)
                percentage += criterion_dissipativity(x_pred, model, return_percentage=True)

            loss_mse /= len(dataloader_test)
            loss_dissipativity /= len(dataloader_test)
            percentage /= len(dataloader_test)
            print(f'Test MSE Loss: {loss_mse.item()}')
            print(f'Test Dissipativity Loss: {loss_dissipativity.item()}')
            print(f'Test Dissipativity Violation Percentage: {percentage.item()}')
            mlflow.log_metric('test/mse_loss_eval', loss_mse.item())
            mlflow.log_metric('test/dissipativity_loss_eval', loss_dissipativity.item())
            mlflow.log_metric('test/dissipativity_violation_percentage_eval', percentage.item())

        # Simulate a trajectory and test the model on it:
        for i, (x0, _) in enumerate(dataloader_test):
            if i == 10:
                break
        x0 = x0.to(device)[0].unsqueeze(0)
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
        # plt.title('Node 1 Trajectory', fontsize=title_fontsize)
        plt.tick_params(axis='x', labelsize=axis_tick_fontsize)
        plt.tick_params(axis='y', labelsize=axis_tick_fontsize)
        plt.tight_layout()
        # plt.show()

        plt.savefig('Node 1 Trajectory.svg', format='svg')
        mlflow.log_artifact('Node 1 Trajectory.svg', "plots")
        os.remove('Node 1 Trajectory.svg')
        plt.close()

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
        # plt.title('Node 1 Dissipativity Violations', fontsize=title_fontsize)
        plt.tick_params(axis='x', labelsize=axis_tick_fontsize)
        plt.tick_params(axis='y', labelsize=axis_tick_fontsize)
        plt.tight_layout()
        # plt.show()

        plt.savefig('Node 1 Dissipativity Violations.svg', format='svg')
        mlflow.log_artifact('Node 1 Dissipativity Violations.svg', "plots")
        os.remove('Node 1 Dissipativity Violations.svg')
        plt.close()

        # Plot dissipativity loss over time for node 1:
        plt.axhline(y=0, color='red', linestyle='--')
        plt.plot(t.cpu().numpy(), dissipativity_loss.cpu().numpy(), label='Dissipativity Loss', color='blue')
        plt.xlabel('Time', fontsize=axis_label_fontsize)
        plt.ylabel('$s(u,y)-\\dot{V}(x)$', fontsize=axis_label_fontsize)
        # plt.title('Node 1 Dissipativity Loss', fontsize=title_fontsize)
        plt.tick_params(axis='x', labelsize=axis_tick_fontsize)
        plt.tick_params(axis='y', labelsize=axis_tick_fontsize)
        plt.tight_layout()
        # plt.show()

        plt.savefig('Node 1 Dissipativity Loss.svg', format='svg')
        mlflow.log_artifact('Node 1 Dissipativity Loss.svg', "plots")
        os.remove('Node 1 Dissipativity Loss.svg')
        plt.close()

        # Plot storage function over time for node 1:
        storage = dissipativity.evaluate_storage(x_pred.squeeze())[:, 0]
        plt.plot(t.cpu().numpy(), storage.cpu().numpy(), label='Storage Function', color='blue')
        plt.xlabel('Time', fontsize=axis_label_fontsize)
        plt.ylabel('V(x)', fontsize=axis_label_fontsize)
        # plt.title('Node 1 Storage Function', fontsize=title_fontsize)
        plt.tick_params(axis='x', labelsize=axis_tick_fontsize)
        plt.tick_params(axis='y', labelsize=axis_tick_fontsize)
        plt.tight_layout()
        # plt.show()

        plt.savefig('Node 1 Storage Function.svg', format='svg')
        mlflow.log_artifact('Node 1 Storage Function.svg', "plots")
        os.remove('Node 1 Storage Function.svg')
        plt.close()
