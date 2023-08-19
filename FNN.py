import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import SGD # resp. Adam
from torch.nn.init import xavier_uniform_
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

# custom dataset definition for parameter-to-solution mapping
class ParamSolMapDataset(Dataset):
    # initialize the dataset
    def __init__(self, train_test_examples_csv):
        # load the csv file as a dataframe
        df = pd.read_csv(train_test_examples_csv, header=None)
        # store the inputs (=parameters) and outputs (=solution of parametrized optimization problem)
        self.X = df.values[:, [0]].astype('float32')
        self.y = df.values[:, 1:3].astype('float32')

    # number of rows/examples in the dataset, since one row describes one example (s. Section 2.2.1. of my thesis)
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # split the dataset into training and test set
    def get_splits(self, test_ratio=0.33): 
        # set the sizes of the training and test set
        test_size = round(test_ratio * len(self.X))
        train_size = len(self.X) - test_size
        # randomly split the dataset according to the desired ratio
        return random_split(self, [train_size, test_size])


# FNN architecture for approximating parameter-to-solution mapping
class FNN(nn.Sequential):
    # initialize FNN
    def __init__(self, layers, activations, xavier_weights=True):
        """
        Arguments:
            layers: list of layer sizes, including input and output sizes
            activations: list of activation functions for each hidden layer
            
        Example:
            model = FNN(layers=[input_size, hiddenLayer1_size, outputsize],
                        activations=[nn.ReLU(), nn.Sigmoid()])
        """
        super(FNN, self).__init__()  
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            linear_layer = nn.Linear(layers[i], layers[i + 1])
            if xavier_weights:
                xavier_uniform_(linear_layer.weight)
            self.layers.append(linear_layer)
            if i < len(activations):
                self.layers.append(activations[i])

    # defines how to forward propagate input through the model
    def forward(self, X):
        # input to 1st hidden layer
        for layer in self.layers:
            X = layer(X)
        return X

# the following methods will help us to supply the network with data and to process them accordingly

# splits the data set into training and test data sets (in our case 70:30) and saves them separately in a csv file
def process_and_save_data(train_test_examples_csv, test_split_size, dataset_size, opt_problem_spec, model_no):
    # load the dataset
    dataset = ParamSolMapDataset(train_test_examples_csv)
    # calculate split
    train, test = dataset.get_splits(test_split_size)
    # prepare data loaders
    train_data = DataLoader(train, batch_size=32, shuffle=True)
    test_data = DataLoader(test, batch_size=1024, shuffle=False)
    # process and save the generated test set
    test_set = []
    for batch_idx, (inputs, targets) in enumerate(test_data):
        print(inputs.shape, " ", targets.shape)
        test_set.append(np.hstack((inputs.numpy(), targets.numpy())))
    tests = np.vstack(test_set)
    np.savetxt(f"SGD_testset_{dataset_size}_mozart_{opt_problem_spec}_{model_no}_set.csv", tests)
    # process and save the generated train set
    train_set = []
    for batch_idx, (inputs, targets) in enumerate(train_data):
        train_set.append(np.hstack((inputs.numpy(), targets.numpy())))
    trains = np.vstack(train_set)
    np.savetxt(f"SGD_trainset_{dataset_size}_mozart_{opt_problem_spec}_{model_no}_set.csv", trains)

    return train_data, test_data


# returns both the network output and the targets for the training and test sets
def run_model(model, test_loader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, targets_batch in test_loader:
            inputs = inputs.to(device)
            targets_batch = targets_batch.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy().squeeze())
            targets.append(targets_batch.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    return predictions, targets


# prints the errors for the training and test sets, 
# stores the losses for these two sets and the scores in a file with two columns each for training and test cases;
# the losses are returned
def train_and_evaluate_model(train_data, test_data, model, learning_rate, dataset_size, opt_problem_spec, model_no, epochs=1500):
    # set loss function and optimization algorithm
    criterion = nn.MSELoss(reduction='mean')
    optimizer = SGD(model.parameters(), learning_rate, momentum=0.9)
    # resp.: optimizer = Adam(model.parameters(), learning_rate)

    # numpy array to save all losses and R2_Scores per epoch
    epoch_losses = np.zeros((epochs, 2))
    epoch_R2s = np.zeros((epochs, 2))
    losses_total = np.zeros((epochs, 4))

    # enumerate epochs/number of batches, same training set several times until weights and biases are optimized
    for epoch in tqdm(range(epochs)):

        # train model on train set
        
        train_losses = []
        train_errors = []
        R2_values_train = []
        for i, (X, Y) in enumerate(train_data):
            # clear the gradients
            optimizer.zero_grad()  
            # compute the model output, in model: forward function
            Y_ = model(X)  
            # calculate loss
            loss = criterion(Y, Y_)
            # backwardpropagation through the model
            loss.backward()  
            # update model weights
            optimizer.step()
            with torch.no_grad():
                percentage_train = r2_score(Y, Y_)
            train_losses.append(loss.item())  # for every batch
            train_errors.append(torch.mean(torch.abs(Y_ - Y)).item())
            R2_values_train.append(percentage_train.item())

        train_loss = np.mean(train_losses) 
        train_error = np.mean(train_errors)
        R2_value_train = np.mean(R2_values_train)

        
        # evaluate trained model on test set
        
        test_losses = []
        test_errors = []
        R2_values_test = []
        # to avoid calling backward
        with torch.no_grad():  
            for i, (X, Y) in enumerate(test_data):
                # compute the model output, in model: forward function
                Y_ = model(X)  
                loss = criterion(Y, Y_)
                test_losses.append(loss.item())
                errors = torch.abs(Y_ - Y)
                test_errors.append(torch.mean(torch.abs(Y_ - Y)).item())
                R2_values_test.append(r2_score(Y, Y_).item())

            test_loss = np.mean(test_losses)
            test_error = np.mean(test_errors)
            R2_value_test = np.mean(R2_values_test)

        epoch_losses[epoch] = [train_loss, test_loss]
        epoch_R2s[epoch] = [R2_value_train, R2_value_test]
        losses_total[epoch] = [train_loss, test_loss, R2_value_train, R2_value_test]

    print(f"Train Error: {train_error} and Test Error: {test_error}")
    np.savetxt(f"SGD_train_and_test_losses_{dataset_size}_mozart_{opt_problem_spec}_{model_no}_{learning_rate}_set.txt", epoch_losses, delimiter=",")
    np.savetxt(f"SGD_R2_train_and_test_scores_{dataset_size}_mozart_{opt_problem_spec}_{model_no}_{learning_rate}_set.txt", epoch_R2s, delimiter=",")
    return losses_total
