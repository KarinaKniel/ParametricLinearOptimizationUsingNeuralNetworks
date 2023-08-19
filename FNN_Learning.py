import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from FNN import FNN
import FNN as fnn

x_dim = 2
p_dim = 1
dataset_size = 500
opt_problem_spec = "c1" # c = (c1 + p, c2)
model_no = 1

dataset = 'mozart_500_c_set.csv' # dataset of the mozart problem of size 500 for the parametrized optimization problem, where c = (c1 + p, c2)
test_split_size = 0.33
n_inputs = p_dim
n_outputs = x_dim
print(x_dim)
model_layers = [n_inputs, 2, 10, n_outputs]
model_activations = [nn.ReLU(), nn.ReLU()]
learning_rates = np.logspace(-4, -1, 1)
log_plot = True

train_data, test_data = fnn.process_and_save_data(dataset, test_split_size, dataset_size, opt_problem_spec, model_no)
print(len(train_data.dataset), len(test_data.dataset))

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
ax[0].set_title("Train")
ax[1].set_title("Test")

for lr in learning_rates:
    print(f"learning rate: {lr}")
    # define the network
    model = FNN(model_layers, model_activations)
    loss = fnn.train_and_evaluate_model(train_data, test_data, model, lr, dataset_size, opt_problem_spec, model_no)
    ax[0].plot(loss[:, 0], label=lr)
    ax[1].plot(loss[:, 1], label=lr)

    # save the weights
    torch.save(model.state_dict(), f'SGD_model_weights_{lr}_{dataset_size}_transport_{opt_problem_spec}_{model_no}_set.pth')

test_predictions_all = {}

for lr in learning_rates:
    # reload the model with the saved weights
    model = FNN(model_layers, model_activations)
    model.load_state_dict(torch.load(f'SGD_model_weights_{lr}_{dataset_size}_transport_{opt_problem_spec}_{model_no}_set.pth'))
    print(lr)

    # make test predictions
    test_predictions, test_targets = fnn.run_model(model, test_data, device)
    test_predictions_all[lr] = test_predictions

for lr, test_predictions in test_predictions_all.items():
    np.savetxt(f'SGD_test_predictions_{lr}_{dataset_size}_transport_{opt_problem_spec}_{model_no}_set.csv', test_predictions,
               delimiter=', ')
    np.savetxt(f'SGD_test_data_{lr}_{dataset_size}_transport_{opt_problem_spec}_{model_no}_set.csv', test_targets, delimiter=',')

train_predictions_all = {}

for lr in learning_rates:
    # reload the model with the saved weights
    model = FNN(model_layers, model_activations)
    model.load_state_dict(torch.load(f'SGD_model_weights_{lr}_{dataset_size}_transport_{opt_problem_spec}_{model_no}_set.pth'))
    print(lr)

    # make training predictions
    train_predictions, train_targets = fnn.run_model(model, train_data, device)
    train_predictions_all[lr] = train_predictions

    # save training predictions
    np.savetxt(f'SGD_train_predictions_{lr}_{dataset_size}_transport_{opt_problem_spec}_{model_no}_set.csv', train_predictions,
               delimiter=', ')

    # save training targets
    np.savetxt(f'SGD_train_data_{lr}_{dataset_size}_transport_{opt_problem_spec}_{model_no}_set.csv', train_targets, delimiter=',')

ax[0].legend()
ax[1].legend()
if log_plot:
    ax[0].set_yscale('log')
ax[1].set_yscale('log')
plt.tight_layout()
plt.show()
