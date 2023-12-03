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
from get_equity import *
from network_tools_v2 import *
from demand_synthesizer import synthesize_demand
from IPython.display import clear_output

#%%
filename = "CampisanoNet2_MOD_PUMP.inp"
valve_locations = ['4','5','7','17','26','29']
filename = make_valve(filename, valve_locations)
#%% Demand Assignment
network = wntr.network.WaterNetworkModel(filename)
n_dem_nodes = len(network.junction_name_list) 
np.random.seed(63)
demand_mults =  np.random.uniform(0.3, 2, n_dem_nodes)
# demand_mults = [2,2,2,2,2,2,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.01,0.01,0.01,0.01,0.01,0.01]
filename = change_demands(filename, demand_mults)
print(filename.stem)

#%% Convert to CV-Tank
CVTank = True
if CVTank:  
    filename = to_CVTank(filename, 35, 10)
    get_equity = get_equity_mix
else: get_equity = get_equity_PDA

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
optimizer = ng.optimizers.DiscreteOnePlusOne(parametrization=param, budget=500, num_workers=4)

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


#%%
filename = "CampisanoNet2_MOD.inp"
valve_locations = ['4','5','7','17','26','29']
filename = make_valve(filename, valve_locations)
network = wntr.network.WaterNetworkModel(filename)
n_dem_nodes = len(network.junction_name_list) 
param = ng.p.TransitionChoice([0, 200, 600, 2000, 5000, 10000], repetitions=num_valves)
optimizer = ng.optimizers.NGOpt(parametrization=param, budget=50, num_workers=4)
# demand_mults = [2,2,2,2,2,2,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.01,0.01,0.01,0.01,0.01,0.01]
print(filename.stem)
biggest_difference = 0 
for i in range(200):
    print("SEED NO. ", i)
    np.random.seed(i)
    demand_mults =  np.random.uniform(0.3, 2, n_dem_nodes)
    filename2 = change_demands(filename, demand_mults)
    filename2 = to_CVTank(filename2, 33, 23)
    get_equity = get_equity_mix
    num_valves = len(network.tcv_name_list)
    null_solution = (0,) * num_valves
    baseline_equity , baseline_asr = 1 - get_equity(filename2, null_solution)[0], get_equity(filename2, null_solution)[1]
    incumbent  = baseline_equity

    recommendation = optimizer.provide_recommendation()

    for _ in range(optimizer.budget):
        x = optimizer.ask()
        loss, dump = get_equity(filename2, x.value, supress_output=True)
        if 1-loss > incumbent:
            print("Incumbent Equity is {:3.3}% for setting {}".format((1-loss)*100, x.value))
            incumbent = 1 - loss
        optimizer.tell(x, loss)
    
    recommendation = optimizer.provide_recommendation()
    optimized_equity = 1 - get_equity(filename2, recommendation.value)[0]
    asr = get_equity(filename2, recommendation.value)[1]

    difference = optimized_equity - baseline_equity
    if difference > biggest_difference:
        biggest_difference = difference
        biggest_seed = i
    print("Null Solution yield Equity of {:3.3}% vs. optimized solution of {:3.3}%".format(baseline_equity*100,optimized_equity*100))
    print("Null setting is {}".format(null_solution))
    print("Optimized Setting is {}".format(recommendation.value))

    print("ASR value is {:3.3} vs baseline value {:3.3}".format(asr,baseline_asr))
    # clear_output(True)

print(" BIGGEST DIFFERENCE AT SEED {} IS {}".format(i, biggest_difference * 100))


#%%
filename = "CampisanoNet2_MOD_PUMP.inp"
valve_locations = ['4','5','7','17','26','29']
filename = make_valve(filename, valve_locations)

param = ng.p.TransitionChoice([0, 200, 600, 2000, 5000, 10000], repetitions=num_valves)
optimizer = ng.optimizers.DiscreteOnePlusOne(parametrization=param, budget=100, num_workers=4)

network = wntr.network.WaterNetworkModel(filename)
num_valves = len(network.tcv_name_list)
n_dem_nodes = len(network.junction_name_list) 

demand_mults = synthesize_demand(n_dem_nodes,5)

comparison_dataframe = pd.DataFrame()

for index, row in demand_mults.iterrows():
    current_mults = list(row)
    filename2 = change_demands(filename, current_mults)

    filename2 = to_CVTank(filename2, 27, 23)
    get_equity = get_equity_mix

    num_valves = len(network.tcv_name_list)
    null_solution = (0,) * num_valves
    baseline_equity , baseline_asr = 1 - get_equity(filename2, null_solution)[0], get_equity(filename2, null_solution)[1]
    incumbent  = baseline_equity

    recommendation = optimizer.provide_recommendation()

    for _ in range(optimizer.budget):
        x = optimizer.ask()
        loss, dump = get_equity(filename2, x.value, supress_output=True)
        if 1-loss > incumbent:
            print("Incumbent Equity is {:3.3}% for setting {}".format((1-loss)*100, x.value))
            incumbent = 1 - loss
            best = x.value
        optimizer.tell(x, loss)
    
    recommendation = optimizer.provide_recommendation()
    optimized_equity = 1 - get_equity(filename2, best)[0]
    asr = get_equity(filename2, best)[1]
    print("Null Solution yield Equity of {:3.3}% vs. optimized solution of {:3.3}%".format(baseline_equity*100,optimized_equity*100))
    print("Null setting is {}".format(null_solution))
    print("Optimized Setting is {}".format(recommendation.value))

    print("ASR value is {:3.3} vs baseline value {:3.3}".format(asr,baseline_asr))

    comparison_dataframe.at[index,"Baseline"] = baseline_equity * 100
    comparison_dataframe.at[index,"Optimized"] = optimized_equity * 100

comparison_dataframe['Difference'] = comparison_dataframe["Optimized"] - comparison_dataframe["Baseline"]


    