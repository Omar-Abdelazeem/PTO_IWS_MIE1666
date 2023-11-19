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

 # Replace with appropriate path and filename
directory=pathlib.Path("")
filename=pathlib.Path("CampisanoNet2_CV-Tank_2valves.inp")
name_only=str(filename.stem)
print("Selected File: ",name_only)
path=directory/filename

# create network model from input file
network = wntr.network.WaterNetworkModel(filename)
list_of_valves = ['7','17']
for pipe in network.pipes():
    for valve_loc in list_of_valves:
        if pipe[0] == valve_loc:
            end_node_i = network.get_node(pipe[1].end_node)
            network.add_junction(str(end_node_i)+"INT", elevation = network.get_node(end_node_i).elevation, coordinates = end_node_i.coordinates)
            network.add_valve("TCV"+pipe[0],str(end_node_i)+"INT", pipe[1].end_node_name,0.4,"TCV",0,0)
            pipe[1].end_node = network.get_node(str(end_node_i)+"INT")
            

path2 = path.parent / (name_only+"_"+str(len(list_of_valves))+'valves.inp')
wntr.epanet.io.InpFile().write(path2, wn = network)