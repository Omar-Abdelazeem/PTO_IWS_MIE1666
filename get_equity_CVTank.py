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

def get_equity(filename, setting):
    # Replace with appropriate path and filename
    directory=pathlib.Path("")
    filename=pathlib.Path(filename)
    name_only=str(filename.stem)
    path=directory/filename

    # create network model from input file
    network = wntr.network.WaterNetworkModel(path)
    i = 0
    print(setting)
    for tcv in network.tcvs():
        network.get_link(tcv[0]).initial_setting = setting[i]
        i+=1
    ## Extract Supply Duration from .inp file
    supply_duration=int(network.options.time.duration)    # in minutes

    # run simulation
    sim = wntr.sim.EpanetSimulator(network)
    # store results of simulation
    results=sim.run_sim()

    timesrs=pd.DataFrame()
    reporting_time_step=network.options.time.report_timestep
    timesrs[0]=results.node['pressure'].loc[0,:]
    for i in range(reporting_time_step,supply_duration+1,reporting_time_step):
        # Extract Node Pressures from Results
        timesrs=pd.concat([timesrs,results.node['pressure'].loc[i,:]],axis=1)

    # Transpose DataFrame such that indices are time (sec) and Columns are each Node
    timesrs=timesrs.T
    # Filter DataFrame for Columns that contain Data for demand nodes only i.e., Tanks in STM
    timesrs=timesrs.filter(regex='Tank\D+',axis=1)

    # Intialize Series for storing statistics
    mean=pd.Series(dtype='float64')
    median=pd.Series(dtype='float64')
    low_percentile=pd.Series(dtype='float64')
    high_percentile=pd.Series(dtype='float64')

    # Set the percentile values to be calculated
    low_percent_val=10   # Range 0 to 100 ONLY
    high_percent_val=90  # Range 0 to 100 ONLY

    # Loop over each row (time step) in the results and calculate values of mean, median, low and high percentiles
    for row in timesrs.index:
        mean.loc[row]=np.mean(timesrs.loc[row,:])*100
        low_percentile.loc[row]=np.percentile(timesrs.loc[row,:],low_percent_val)*100
        median.loc[row]=np.percentile(timesrs.loc[row,:],50)*100
        high_percentile.loc[row]=np.percentile(timesrs.loc[row,:],high_percent_val)*100

    end_state = timesrs.iloc[-1]
    ASR = np.mean(end_state)
    ADEV = 0
    for user in end_state:
        ADEV += abs(user - ASR)
    ADEV = ADEV / len(end_state)
    UC = 1 - ADEV / ASR
    print("Uniformity Coefficient is {}".format(UC))
    return 1-UC

filename = "CampisanoNet2_CV-Tank_2valves_1.inp"
network = wntr.network.WaterNetworkModel(filename)
num_valves = len(network.tcv_name_list)
null_solution = (0,) * num_valves
baseline_equity = 1 - get_equity(filename, null_solution)

param = ng.p.TransitionChoice([0, 50, 100, 200, 10000], repetitions=num_valves)
optimizer = ng.optimizers.NGOpt(parametrization=param, budget=20, num_workers=1)

recommendation = optimizer.provide_recommendation()

for _ in range(optimizer.budget):
    x = optimizer.ask()
    # loss = onemax(*x.args, **x.kwargs)  # equivalent to x.value if not using Instrumentation
    loss = get_equity(filename, x.value)
    optimizer.tell(x, loss)

recommendation = optimizer.provide_recommendation()
optimized_equity = 1 - get_equity(filename, recommendation.value)

print("Null Solution yield Equity of {:3.3}% vs. optimized solution of {:3.3}%".format(baseline_equity*100,optimized_equity*100))
print("Null setting is {}".format(null_solution))
print("Optimized Setting is {}".format(recommendation.value))

