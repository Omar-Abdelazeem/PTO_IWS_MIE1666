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
from network_tools2 import *
from demand_synthesizer import synthesize_demand
from IPython.display import clear_output

#%%
# filename = "Network2_12hr_PDA.inp"
# valve_locations = ['12','22','18','7','52','74','37','53']

filename = "CampisanoNet2_MOD_PUMP.inp"
valve_locations = ['4','5','7','17','26','29']

filename = make_valve(filename, valve_locations)

network = wntr.network.WaterNetworkModel(filename)
num_valves = len(network.tcv_name_list)
n_dem_nodes = len(network.junction_name_list) 
param = ng.p.TransitionChoice([0, 200, 600, 2000, 5000, 10000], repetitions=num_valves)
optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=param, budget=100, num_workers=4)

demand_mults = synthesize_demand(n_dem_nodes,5, seed = 100)

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
    print("Null setting is {} vs. Optimized Setting is {}".format(null_solution, best))

    # print("ASR value is {:3.3} vs baseline value {:3.3}".format(asr,baseline_asr))

    comparison_dataframe.at[index,"Baseline"] = baseline_equity * 100
    comparison_dataframe.at[index,"Optimized"] = optimized_equity * 100

comparison_dataframe['Difference'] = comparison_dataframe["Optimized"] - comparison_dataframe["Baseline"]

#%%
comparison_dataframe.to_csv(filename.stem+'Results.csv')