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

def get_equity(filename, setting, plot = False):
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


    if plot:
        supply_duration_hr=supply_duration/3600
        xaxis=np.arange(0,supply_duration_hr+0.00001,reporting_time_step/3600)

        fig, ax=plt.subplots()
        # Change figure size (and aspect ratio) by adjusting height and width here
        fig.set_figwidth(1.5)
        fig.set_figheight(1)

        # Formatting Plot: Setting a Title, x- and y-axis limits, major and minor ticks
        ax.set_title('Distribution of Demand Satisfaction')
        ax.set_xlim(0,supply_duration_hr)
        ax.set_ylim(0,max(median))
        ax.set_xticks(np.arange(0,supply_duration_hr+1,4))
        ax.set_xticks(np.arange(0,supply_duration_hr+1,1),minor=True)
        ax.set_yticks(np.arange(0,max(median),25))
        ax.tick_params(width=0.5)


        # Data to be plotted: Mean as a percentage (hence the multiplication by 100)
        # Change color by changing the string next to c= and linewidth by value
        line1,=ax.plot(xaxis,median, c='#d73027',linewidth=0.5)
        plt.fill_between(xaxis, y1=low_percentile, y2=high_percentile, alpha=0.4, color='#d73027', edgecolor=None)
        plt.xlabel('Supply Time (hr)')
        plt.ylabel('Satisfaction Ratio (%)')
        # Optional: show grid or turn on and off spines (Plot box sides)
        # ax.grid(visible=True,which='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show
    return (1-UC)*ASR, ASR

