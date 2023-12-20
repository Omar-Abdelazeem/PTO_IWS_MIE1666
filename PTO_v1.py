#%% Importing Required Libraries and Packages
import wntr  
import numpy as np 
import pandas as pd
import matplotlib as mpl
from matplotlib import figure
import matplotlib.pyplot as plt
import timeit 
import re
import pathlib
import nevergrad as ng
from get_equity import *
from network_tools_v2 import *
from demand_synthesizer import *
from IPython.display import clear_output

import torch as torch
import torch.nn as nn
import torch_geometric.nn as geom_nn

from torch_geometric.utils import from_networkx
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCN, GAT, GraphSAGE

from NN_tools import *

#%% Input Cell

# Input File Name and Specify Valve Locations (Pipe IDs on which valves will be places)
# filename = "Network2_12hr_PDA.inp"
# valve_locations = ['12','22','18','7','52','74','37','53']

filename = "CampisanoNet2_MOD_PUMP.inp"
valve_locations = ['4','5','7','17','26','29']

# Edit the EPANET file to add the specified valves
filename = make_valve(filename, valve_locations)

# Load the network model
network = wntr.network.WaterNetworkModel(filename)
# Get the number of valves from the network model (failsafe in case some valves failed to create in above step)
num_valves = len(network.tcv_name_list)
# Get the total number of demand nodes in the network
n_dem_nodes = len(network.junction_name_list) 

n_train_days = 7
n_validation_days = 1
n_val_cummulative = n_train_days + n_validation_days
n_test_days = 1
n_test_cummulative = n_val_cummulative + n_test_days

total_days = n_train_days + n_validation_days + n_test_days



#%% Model Training Cell

demand_data = synthesize_demands(n_dem_nodes, total_days, 5, 100, True)
scaled_demand = MinMaxScaler().fit_transform(demand_data)

train = scaled_demand[0:n_train_days * 24, :]
val = scaled_demand[n_train_days * 24 : n_val_cummulative * 24, :]
test = scaled_demand[ n_val_cummulative * 24 : n_test_cummulative * 24, :]

look_back = 1  # Adjust as needed
X_train, y_train = create_dataset_multivariate(train, look_back)
X_val, y_val = create_dataset_multivariate(val, look_back)
X_test, y_test = create_dataset_multivariate(test, look_back)

# Convert to PyTorch tensors
X_train_tensors = torch.tensor(X_train, dtype=torch.float)
y_train_tensors = torch.tensor(y_train, dtype=torch.float)
X_val_tensors = torch.tensor(X_val, dtype=torch.float)
y_val_tensors = torch.tensor(y_val, dtype=torch.float)
X_test_tensors = torch.tensor(X_test, dtype=torch.float)
y_test_tensors = torch.tensor(y_test, dtype=torch.float)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pass through the linear layer
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

# Model parameters
input_dim = X_train_tensors.shape[2]  # Number of features
hidden_dim = 50  # Number of features in the hidden state
output_dim = y_train_tensors.shape[1]  # Output dimension
num_layers = 1   # Number of stacked LSTM layers

# Instantiate the model
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers) # Change depending on the model we want to use

# Creating datasets from tensors
train_dataset = TensorDataset(X_train_tensors, y_train_tensors)
val_dataset = TensorDataset(X_val_tensors, y_val_tensors)

# Creating DataLoaders for batching the data
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#%% Train The Model
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    # Training loop
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_running_loss = 0.0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

    # Display the epoch and loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Training Loss: {running_loss/len(train_loader):.4f}, '
              f'Validation Loss: {val_running_loss/len(val_loader):.4f}')

# Set model to evaluation mode
model.eval()

# Predictions with no gradient calculations
with torch.no_grad():
    # Predict for training, validation, and testing sets
    train_predict = model(X_train_tensors).cpu().numpy()
    val_predict = model(X_val_tensors).cpu().numpy()
    test_predict = model(X_test_tensors).cpu().numpy()

# Inverse transform the predictions to original scale
train_predict = MinMaxScaler().inverse_transform(train_predict)
val_predict = MinMaxScaler().inverse_transform(val_predict)
test_predict = MinMaxScaler().inverse_transform(test_predict)

# Inverse transform the original training, validation, and testing data
y_train_original = MinMaxScaler().inverse_transform(y_train)
y_val_original = MinMaxScaler().inverse_transform(y_val)
y_test_original = MinMaxScaler().inverse_transform(y_test)

# Calculate MSE
train_mse = mean_squared_error(y_train_original, train_predict)
val_mse = mean_squared_error(y_val_original, val_predict)
test_mse = mean_squared_error(y_test_original, test_predict)

# Calculate MAE
train_mae = mean_absolute_error(y_train_original, train_predict)
val_mae = mean_absolute_error(y_val_original, val_predict)
test_mae = mean_absolute_error(y_test_original, test_predict)

# Calculate R^2
train_r2 = r2_score(y_train_original, train_predict)
val_r2 = r2_score(y_val_original, val_predict)
test_r2 = r2_score(y_test_original, test_predict)

# Calculate RMSE
train_rmse = np.sqrt(train_mse)
val_rmse = np.sqrt(val_mse)
test_rmse = np.sqrt(test_mse)

# Print metrics with headers and alignment
print("Metric\t\tTrain Set\tValidation Set\tTest Set")
print(f"MSE\t\t{train_mse:.7f}\t{val_mse:.7f}\t{test_mse:.7f}")
print(f"RMSE\t\t{train_rmse:.7f}\t{val_rmse:.7f}\t{test_rmse:.7f}")
print(f"MAE\t\t{train_mae:.7f}\t{val_mae:.7f}\t{test_mae:.7f}")
print(f"R^2\t\t{train_r2:.7f}\t{val_r2:.7f}\t{test_r2:.7f}")


#%% Optimization Cell
# Parameterization for the optimizer: set the discrete space of possible valve settings
param = ng.p.TransitionChoice([0, 200, 600, 2000, 5000, 10000], repetitions=num_valves)
# Initialize the optimizer with specified maximum iterations
optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=param, budget=100, num_workers=4)

# TEMP: synthesize 1 day's demand for n_dem_nodes nodes
demand_mults = synthesize_demand(n_dem_nodes,5, seed = 100)

# Dataframe to store results comparing baseline (null solution, no optimization)
# to the optimized solution
comparison_dataframe = pd.DataFrame()

# Loop over each timestep in the time period
for index, row in demand_mults.iterrows():
    # get a list of demand multipliers for the current time step
    current_mults = list(row)
    # apply the demand multipliers to the input file
    filename2 = change_demands(filename, current_mults)

    # Convert the simulation into an IWS simulation (See Abdelazeem and Meyer 2023)
    filename2 = to_CVTank(filename2, 27, 23)
    # initialize the get_equity function
    get_equity = get_equity_mix

    # null solution is leaving all the valves fully open (0 resistance)
    null_solution = (0,) * num_valves
    # get baseline equity and satisfaction ratio using null solution
    baseline_equity , baseline_asr = 1 - get_equity(filename2, null_solution)[0], get_equity(filename2, null_solution)[1]
    # store the best equity value
    incumbent  = baseline_equity

    # suggest a point to explore
    recommendation = optimizer.provide_recommendation()

    # For number of iterations <= budget
    for _ in range(optimizer.budget):
        # Suggest a point to explore
        x = optimizer.ask()
        # get the equity of that x
        loss, dump = get_equity(filename2, x.value, supress_output=True)
        # if the equity is better than the best equity
        if 1-loss > incumbent:
            # Update best equity and best setting
            print("Incumbent Equity is {:3.3}% for setting {}".format((1-loss)*100, x.value))
            incumbent = 1 - loss
            best = x.value
        # Feedback to optimizer the value of the evaluated x
        optimizer.tell(x, loss)
    
    recommendation = optimizer.provide_recommendation()
    # The equity of the optimal solution
    optimized_equity = 1 - get_equity(filename2, best)[0]
    # The ASR of the best solution
    asr = get_equity(filename2, best)[1]
    print("Null Solution yield Equity of {:3.3}% vs. optimized solution of {:3.3}%".format(baseline_equity*100,optimized_equity*100))
    print("Null setting is {} vs. Optimized Setting is {}".format(null_solution, best))

    # print("ASR value is {:3.3} vs baseline value {:3.3}".format(asr,baseline_asr))

    # Store results
    comparison_dataframe.at[index,"Baseline"] = baseline_equity * 100
    comparison_dataframe.at[index,"Optimized"] = optimized_equity * 100

comparison_dataframe['Difference'] = comparison_dataframe["Optimized"] - comparison_dataframe["Baseline"]

#%%
comparison_dataframe.to_csv(filename.stem+'Results.csv')