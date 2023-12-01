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
from IPython.display import clear_output

def synthesize_demands(n_nodes, n_days, n_components, seed, plot = False):
    '''
    n_nodes (int): number of nodes to generate demands for
    n_days (int): how long (in days) to generate demands for (repeated units of one day)
    n_components (int): how many patterns from the dataset to use when synthesizing a new pattern
    '''
    
    demand_patterns = pd.read_csv("DEMAND_PATTERNS.csv")
    demand_patterns.set_index("TIME", inplace =True)

    assert n_components <= len(demand_patterns.columns), " Enter a proper n_components < n demand patterns in dataset"

    np.random.seed(seed)
    new_demands = pd.DataFrame()
    for node in range(n_nodes):
        sample_patterns = demand_patterns.sample(axis = 1, n = n_components)
        alphas=np.random.uniform(0.1, 1.5, n_components)
        new_demand = alphas[0] * sample_patterns.iloc[:,0]
        for column in np.arange(1,n_components):
            new_demand = new_demand + alphas[column]* sample_patterns.iloc[:,column]
        new_demand = new_demand / new_demand.mean()
        new_demand.name = 'Node'+str(node+1)

        shift_amount = np.random.randint(-2,3)
        shifted_demand = new_demand.shift(shift_amount)
        if shift_amount>0:
            shifted_demand.iloc[0:shift_amount] = new_demand.iloc[-shift_amount:]
        elif shift_amount < 0:
            shifted_demand.iloc[shift_amount:] = new_demand.iloc[:-shift_amount]

        new_demands = pd.concat([new_demands,shifted_demand], axis = 1)

    new_demands = pd.concat([new_demands] * n_days, ignore_index= True)
    new_demands.set_index(np.arange(1,len(new_demands)+1), inplace=True)

    if plot:
        fig, ax = plt.subplots()
        for node in range(n_nodes):
            ax.plot(new_demands.index[0:24], new_demands.iloc[0:24,node])
        
        summed_demand = new_demands.sum(axis=1)
        fig, ax = plt.subplots()
        ax.plot(new_demands.index[0:24],summed_demand.iloc[0:24])

    return new_demands


def synthesize_demand(n_nodes, n_components, seed, plot = False):
    '''
    n_nodes (int): number of nodes to generate demands for
    n_components (int): how many patterns from the dataset to use when synthesizing a new pattern
    '''
    
    demand_patterns = pd.read_csv("DEMAND_PATTERNS.csv")
    demand_patterns.set_index("TIME", inplace =True)

    assert n_components <= len(demand_patterns.columns), " Enter a proper n_components < n demand patterns in dataset"

    np.random.seed(seed)

    new_demands = pd.DataFrame()
    for node in range(n_nodes):
        sample_patterns = demand_patterns.sample(axis = 1, n = n_components)
        alphas=np.random.uniform(0.1, 1.5, n_components)
        new_demand = alphas[0] * sample_patterns.iloc[:,0]
        for column in np.arange(1,n_components):
            new_demand = new_demand + alphas[column]* sample_patterns.iloc[:,column]
        new_demand = new_demand / new_demand.mean()
        new_demand.name = 'Node'+str(node+1)

        shift_amount = np.random.randint(-2,3)
        shifted_demand = new_demand.shift(shift_amount)
        if shift_amount>0:
            shifted_demand.iloc[0:shift_amount] = new_demand.iloc[-shift_amount:]
        elif shift_amount < 0:
            shifted_demand.iloc[shift_amount:] = new_demand.iloc[:-shift_amount]

        new_demands = pd.concat([new_demands,shifted_demand], axis = 1)

    return new_demands

#%%
# synthesize_demand(20,3, True)

#%% load data
# demand_patterns = pd.read_csv("DEMAND_PATTERNS.csv")
# demand_patterns.set_index("TIME", inplace =True)
#%%
# n_nodes = 20
# n_days = 7
# n_components = 5

# new_demands = pd.DataFrame()
# for node in range(n_nodes):
#     sample_patterns = demand_patterns.sample(axis = 1, n = n_components)
#     alphas=np.random.uniform(0.1, 1.5, n_components)
#     new_demand = alphas[0] * sample_patterns.iloc[:,0]
#     for column in np.arange(1,n_components):
#         new_demand = new_demand + alphas[column]* sample_patterns.iloc[:,column]
#     new_demand = new_demand / new_demand.mean()
#     new_demand.name = str(node)

#     shift_amount = np.random.randint(-2,3)
#     shifted_demand = new_demand.shift(shift_amount)
#     if shift_amount>0:
#         shifted_demand.iloc[0:shift_amount] = new_demand.iloc[-shift_amount:]
#     elif shift_amount < 0:
#         shifted_demand.iloc[shift_amount:] = new_demand.iloc[:-shift_amount]

#     new_demands = pd.concat([new_demands,shifted_demand], axis = 1)

# fig, ax = plt.subplots()
# for node in range(n_nodes):
#     ax.plot(new_demand.index, new_demands[str(node)])


# new_demands = pd.concat([new_demands] * n_days, ignore_index= True)
# new_demands.set_index(np.arange(1,len(new_demands)+1), inplace=True)

# fig, ax = plt.subplots()
# for node in range(n_nodes):
#     ax.plot(new_demands.index, new_demands[str(node)])
