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
from demand_synthesizer import synthesize_demand
from IPython.display import clear_output
import time

#%% Main Loop

start = time.time()
# Input File Name and Specify Valve Locations (Pipe IDs on which valves will be places)

filename = "Network3_12hr_PDA.inp"
valve_locations = ["291", "308", "221", "58", "34", "19", "1", "126", "142", "123", "154", "163"]

# filename = "Network2_12hr_PDA.inp"
# valve_locations = ['12','22','18','7','52','74','37','53']

# filename = "CampisanoNet2_MOD_PUMP.inp"
# valve_locations = ['4','5','7','17','26','29']

# Edit the EPANET file to add the specified valves
filename = make_valve(filename, valve_locations)

# Load the network model
network = wntr.network.WaterNetworkModel(filename)
# Get the number of valves from the network model (failsafe in case some valves failed to create in above step)
num_valves = len(network.tcv_name_list)
# Get the total number of demand nodes in the network
n_dem_nodes = len(network.junction_name_list) 

# Parameterization for the optimizer: set the discrete space of possible valve settings
param = ng.p.TransitionChoice([0, 200, 600, 2000, 5000, 10000], repetitions=num_valves)
# Initialize the optimizer with specified maximum iterations
optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=param, budget=100, num_workers=4)

# TEMP: synthesize 1 day's demand for n_dem_nodes nodes
demand_mults = synthesize_demand(n_dem_nodes,5, seed = 200)

# Dataframe to store results comparing baseline (null solution, no optimization)
# to the optimized solution
comparison_dataframe = pd.DataFrame()
#%%
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
    best = null_solution
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

execution = time.time() - start
comparison_dataframe['Difference'] = comparison_dataframe["Optimized"] - comparison_dataframe["Baseline"]

#%%
comparison_dataframe.to_csv(filename.stem+'Results.csv')