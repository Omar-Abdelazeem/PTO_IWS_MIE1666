#%%
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
from get_equity_PDA import *
from network_tools_v2 import *

#%%
filename = "CampisanoNet2_4x_uniform.inp"
valve_locations = ['4','5','26','29']
filename = make_valve(filename, valve_locations)

#%% Demand Assignment
network = wntr.network.WaterNetworkModel(filename)
n_dem_nodes = len(network.junction_name_list) 
# np.random.seed(110)
demand_mults =  np.random.uniform(0.5, 2, n_dem_nodes)
# demand_mults = [3,3,3,3,3,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.01,0.01,0.01,0.01,0.01,0.01]
filename = change_demands(filename, demand_mults)
print(filename.stem)

#%% Convert to CV-Tank
# filename = to_CVTank(filename, 2, -1)

#%%
get_equity(filename, (0,) * len(network.tcv_name_list))
#%%
num_valves = len(network.tcv_name_list)
null_solution = (0,) * num_valves
baseline_equity , baseline_asr = 1 - get_equity(filename, null_solution)[0], get_equity(filename, null_solution)[1]
incumbent  = baseline_equity
print("Baseline Equity is {:3.3}%".format(baseline_equity*100))

# instrum = ng.p.Instrumentation(
    # ng.p.Array(shape=num_valves).set_bounds(0,10000)
# )
param = ng.p.TransitionChoice([0, 200, 600, 2000, 5000, 10000], repetitions=num_valves)
optimizer = ng.optimizers.NGOpt(parametrization=param, budget=500, num_workers=4)

recommendation = optimizer.provide_recommendation()

for _ in range(optimizer.budget):
    x = optimizer.ask()
    loss, dump = get_equity(filename, x.value, supress_output=True)
    if 1-loss > incumbent:
        print("Incumbent Equity is {:3.3}% for setting {}".format((1-loss)*100, x.value))
        incumbent = 1 - loss
    optimizer.tell(x, loss)
#%%
recommendation = optimizer.provide_recommendation()
optimized_equity = 1 - get_equity(filename, recommendation.value)[0]
asr = get_equity(filename, recommendation.value)[1]

print("Null Solution yield Equity of {:3.3}% vs. optimized solution of {:3.3}%".format(baseline_equity*100,optimized_equity*100))
print("Null setting is {}".format(null_solution))
print("Optimized Setting is {}".format(recommendation.value))

print("ASR value is {:3.3} vs baseline value {:3.3}".format(asr,baseline_asr))
#%%
a, b = get_equity(filename, (0,0,0,0), True)
print("UC is ", 1- a/b," ASR is ", b)
