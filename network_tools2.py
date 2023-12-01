
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

# directory=pathlib.Path("")
# filename=pathlib.Path("CampisanoNet2_CV-Tank.inp")

def make_valve(filename, list_of_valves):
 # Replace with appropriate path and filename

    directory = pathlib.Path("Net_Files")
    filename=pathlib.Path(filename)
    name_only=str(filename.stem)
    # print("Selected File: ",name_only)
    path=directory / filename
    path = path.resolve()

    # create network model from input file
    network = wntr.network.WaterNetworkModel(path)
    for pipe in network.pipes():
        for valve_loc in list_of_valves:
            if pipe[0] == valve_loc:
                end_node_i = network.get_node(pipe[1].end_node)
                network.add_junction(str(end_node_i)+"INT", elevation = network.get_node(end_node_i).elevation, coordinates = end_node_i.coordinates)
                network.add_valve("TCV"+pipe[0],str(end_node_i)+"INT", pipe[1].end_node_name,0.4,"TCV",0,0)
                pipe[1].end_node = network.get_node(str(end_node_i)+"INT")
                

    path2 = path.parent / (name_only+"_"+str(len(list_of_valves))+'valves.inp')
    wntr.epanet.io.InpFile().write(path2, wn = network)
    return path2

def change_demands(filename, multipliers):
    '''
    Changes the demands in the network based on an ordered vector of demand multipliers applied to base demands
    '''

    directory = pathlib.Path("Net_Files")
    filename=pathlib.Path(filename)
    name_only=str(filename.stem)
    # print("Selected File: ",name_only)
    path=directory / filename
    path = path.resolve()

    # create network model from input file
    network = wntr.network.WaterNetworkModel(path)

    i=0
    for node in network.junction_name_list:
        if "INT" in node:
            break
        # print(node)
        junction = network.get_node(node)
        new_demand = float(multipliers[i]*junction.base_demand)
        junction.demand_timeseries_list[0].base_value = new_demand
        i+=1
    
    wntr.epanet.io.InpFile().write(path, wn = network)
    return filename


def to_CVTank(filename, desired_pressure, minimum_pressure):
    pressure_diff = desired_pressure - minimum_pressure
    directory = pathlib.Path("Net_Files")
    filename=pathlib.Path(filename)
    name_only=str(filename.stem)
    # print("Selected File: ",name_only)
    path=directory / filename
    path = path.resolve()

    demand_nodes=[]       # For storing list of nodes that have non-zero demands
    desired_demands=[]    # For storing demand rates desired by each node for desired volume calculations
    elevations=[]         # For storing elevations of demand nodes
    xcoordinates=[]       # For storing x coordinates of demand nodes
    ycoordinates=[]       # For storing y coordinates of demand nodes
    all_nodes=[]          # For storing list of node ids of all nodes
    all_elevations=[]     # For storing elevations of all nodes
    ## MAYBE SAVE ALL NODE IDS IN DATAFRAME WITH ELEVATION AND BASE DEMAND AND THEN FILTER DATA FRAME LATER FOR DEMAND NODES ONLY

    # Creates a network model object using EPANET .inp file
    network=wntr.network.WaterNetworkModel(path)


    # Get the supply duration in minutes (/60) as an integer
    supply_duration=24

    # Iterates over the junction list in the Network object
    for node in network.junctions():
        # all_nodes.append(node[1].name)
        # all_elevations.append(node[1].elevation)
        # # For all nodes that have non-zero demands
        if node[1].base_demand != 0:
            node_d = node[1]
            vol = round(np.sqrt(node_d.base_demand *60 *supply_duration * 4 / np.pi),4)
            network.add_reservoir('TankforNode'+node_d.name, node_d.elevation + minimum_pressure,None,node_d.coordinates
            )
            l = round(pressure_diff*130**1.852*0.05**4.87/10.67/(node_d.base_demand)**1.852 , 4)
            network.add_pipe('PipeforNode'+node_d.name,node_d.name,'TankforNode'+node_d.name,l,0.05,130,0,'OPEN',True)
            # # Record node ID (name), desired demand (base_demand) in CMS, elevations, x and y coordinates
            # demand_nodes.append(node[1].name)
            # desired_demands.append(node[1].base_demand)
            # elevations.append(node[1].elevation)
            # xcoordinates.append(node[1].coordinates[0])
            # ycoordinates.append(node[1].coordinates[1])

    network.options.hydraulic.demand_multiplier=0.0001
    path2 = pathlib.Path(str(path.parent / path.stem)+'_CVTank.inp')
    wntr.epanet.io.InpFile().write(path2, wn = network)
    print(path2.name)
    return path2.name

#%%
# change_demands (filename, np.random.uniform(0.5, 1.5, 26))



    