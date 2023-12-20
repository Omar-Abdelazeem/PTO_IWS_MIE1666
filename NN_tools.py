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

def create_dataset_multivariate(scaled_demand, look_back=1):
    dataX, dataY = [], []
    for i in range(len(scaled_demand) - look_back):
        a = scaled_demand[i:(i + look_back)]
        dataX.append(a)
        dataY.append(scaled_demand[i + look_back])
    return np.array(dataX), np.array(dataY)