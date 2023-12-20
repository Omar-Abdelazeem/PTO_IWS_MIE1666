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

def get_equity_mix(filename, setting, metric='VC',supress_output = False):
    # Replace with appropriate path and filename
    directory = pathlib.Path("/Users/omaraliamer/Desktop/UofT/PhD/Courses/Fall23/MIE1666/PTO_IWS_MIE1666/Net_Files")
    filename=pathlib.Path(filename)
    name_only=str(filename.stem)
    # print("Selected File: ",name_only)
    path=directory / filename
    path = path.resolve()
    # print(path)

    demand_nodes=[]       # For storing list of nodes that have non-zero demands
    desired_demands=[]    # For storing demand rates desired by each node for desired volume calculations

    # Creates a network model object using EPANET .inp file
    network=wntr.network.WaterNetworkModel(path)

    # Iterates over the junction list in the Network object
    for node in network.junctions():

        # For all nodes that have non-zero demands
        if node[1].base_demand != 0:
            # Record node ID (name) and its desired demand (base_demand) in CMS
            demand_nodes.append(node[1].name)
            desired_demands.append(node[1].base_demand)

    # Get the supply duration in minutes (/60) as an integer
    supply_duration=int(network.options.time.duration/60)

    i = 0
    # print(setting)
    for tcv in network.tcvs():
        network.get_link(tcv[0]).initial_setting = setting[i]
        i+=1
    ## Extract Supply Duration from .inp file

    # run simulation
    sim = wntr.sim.EpanetSimulator(network)
    # store results of simulation
    results=sim.run_sim()


    timesrs_demands=pd.DataFrame(results.link['flowrate'])
    timesrs_demands = timesrs_demands.filter(regex="PipeforNode", axis = 1)

    actual_demands = np.array(timesrs_demands.iloc[-1,:])
    satisfaction = np.divide(actual_demands, np.array(desired_demands)) * 100

    ASR = np.mean(satisfaction)
    first_factor = [Q_act/np.sum(actual_demands) for Q_act in actual_demands]
    second_factor = [Q_req/np.sum(desired_demands) for Q_req in desired_demands]

    VCs = np.array([np.abs(first_f - second_f) for first_f, second_f in zip(first_factor,second_factor)])
    VC_value = 1 - 0.5 * np.sum(VCs)

    ADEV = 0
    for user in satisfaction:
        ADEV += abs(user - ASR)
    ADEV = ADEV / len(satisfaction)
    UC = 1 - ADEV / ASR
    # print("Uniformity Coefficient is {}".format(UC))
    if metric == 'UC':
        objective = 1 - UC
    elif metric == 'VC':
        objective = 1 - VC_value

    else: objective = None
    return objective, ASR


def get_equity_PDA(filename, setting, supress_output = False):
    # Replace with appropriate path and filename
    directory=pathlib.Path("")
    filename=pathlib.Path(filename)
    name_only=str(filename.stem)
    path=directory/filename

    demand_nodes=[]       # For storing list of nodes that have non-zero demands
    desired_demands=[]    # For storing demand rates desired by each node for desired volume calculations

    # Creates a network model object using EPANET .inp file
    network=wntr.network.WaterNetworkModel(filename)

    # Iterates over the junction list in the Network object
    for node in network.junctions():

        # For all nodes that have non-zero demands
        if node[1].base_demand != 0:
            # Record node ID (name) and its desired demand (base_demand) in CMS
            demand_nodes.append(node[1].name)
            desired_demands.append(node[1].base_demand)

    # Get the supply duration in minutes (/60) as an integer
    supply_duration=int(network.options.time.duration/60)

    i = 0
    # print(setting)
    for tcv in network.tcvs():
        network.get_link(tcv[0]).initial_setting = setting[i]
        i+=1
    ## Extract Supply Duration from .inp file

    # run simulation
    sim = wntr.sim.EpanetSimulator(network)
    # store results of simulation
    results=sim.run_sim()

    timesrs_demands=pd.DataFrame()
    timesrs_demands[0]=results.node['demand'].loc[0,:]
    res=results.node['demand']
    for i in range(1,supply_duration+1):
        # Extract Node Pressures from Results
        timesrs_demands=pd.concat([timesrs_demands,results.node['demand'].loc[i*60,:]],axis=1)

    # Transpose DataFrame such that indices are time (sec) and Columns are each Node
    timesrs_demands=timesrs_demands.T
    # Filter DataFrame for Columns that contain Data for demand nodes only
    timesrs_demands=timesrs_demands[demand_nodes]

    # Calculates the total demand volume in the specified supply cycle
    desired_volumes=[]

    # Loop over each desired demand
    for demand in desired_demands:
        # Append the corresponding desired volume (cum) = demand (LPS) *60 sec/min * supply duration (min)
        desired_volumes.append(float(demand)*60*float(supply_duration))

    # Combine demands (LPS) to their corresponding desired volume (cum)
    desired_volumes=dict(zip(demand_nodes,desired_volumes))

    # Initalized DataFrame for storing volumes received by each demand node as a timeseries
    timesrs_satisfaction=pd.DataFrame(index=timesrs_demands.index,columns=desired_volumes.keys())
    # Set Initial volume for all consumers at 0
    timesrs_satisfaction.iloc[0,:]=0

    # Loop over consumers and time steps to add up volumes as a percentage of total desired volume (Satisfaction Ratio)
    for timestep in list(timesrs_satisfaction.index)[1:]:
        for node in timesrs_satisfaction.columns:
            # Cummulatively add the percent satisfaction ratio (SR) increased each time step
            ## SR at time t = SR at time t-1 + demand at time t-1 (cms) *60 seconds per time step/ Desired Demand Volume (cum)
            timesrs_satisfaction.at[timestep,node]=timesrs_satisfaction.at[timestep-60,node]+timesrs_demands.at[timestep-60,node]*60/desired_volumes[node]

    end_state = timesrs_satisfaction.iloc[-1]

    ASR = np.mean(end_state)
    ADEV = 0
    for user in end_state:
        ADEV += abs(user - ASR)
    ADEV = ADEV / len(end_state)
    UC = 1 - ADEV / ASR
    # print("Uniformity Coefficient is {}".format(UC))
    return 1-UC, ASR

#%%
# filename = "CampisanoNet2_4x_uniform_4valves_CVTank.inp"
# filename = pathlib.Path(filename)
# directory = pathlib.Path("Net_Files")
# name_only=str(filename.stem)
# print("Selected File: ",name_only)
# path=directory / filename
# path = path.resolve()
# print(path)
# network = wntr.network.WaterNetworkModel(path)
# num_valves = len(network.tcv_name_list)
# null_solution = (0,) * num_valves
# baseline_equity = 1 - get_equity(filename, null_solution)[0]
# print(baseline_equity)

# param = ng.p.TransitionChoice([0, 50, 100, 200, 10000], repetitions=num_valves)
# optimizer = ng.optimizers.NGOpt(parametrization=param, budget=20, num_workers=1)

# recommendation = optimizer.provide_recommendation()

# for _ in range(optimizer.budget):
#     x = optimizer.ask()
#     # loss = onemax(*x.args, **x.kwargs)  # equivalent to x.value if not using Instrumentation
#     loss = get_equity(filename, x.value)
#     optimizer.tell(x, loss)

# recommendation = optimizer.provide_recommendation()
# optimized_equity = 1 - get_equity(filename, recommendation.value)

# print("Null Solution yield Equity of {:3.3}% vs. optimized solution of {:3.3}%".format(baseline_equity*100,optimized_equity*100))
# print("Null setting is {}".format(null_solution))
# print("Optimized Setting is {}".format(recommendation.value))

