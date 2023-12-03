# Predict-Then-Optimize for Intermittent Water Supply (PTO-IWS) Networks

# Introduction
This repository includes all code and data used to develop a predict-then-optimize framework to optimize valve operation (opening and closing) in order to maximize the equity of water distribution. The following is a breakdown and guide on the constituents of this repository and their description:


## 0. Required libraries and packages
Using this repository requires that the following publicly available packages be installed within the current Python environment:
**WNTR** >= 0.4.0: EPANET's Python Wrapper [1]; used to execute EPANET simulations  
**NeverGrad** > 1.0.0: Gradient Free Black Box Optimization by Meta [2]; used as the optimizer  
**Pandas & Numpy**: for data processing, handling and cleaning  
Additional requirements are outlined in the requirements.txt file  

## 1. Net_Files
This folder contains the EPANET .inp input files that describe the network model (pipes, nodes, tanks, pumps  and their properties, e.g., diameter, length) and can be executed with the EPANET python wrapper.

## 2. demand_synthesizer.py
Our demand pattern generator, generates N distinct demand patterns through randomly combining, scaling and shifting the real demand patterns stored in **DEMAND_PATTERNS.csv**  

## 3. network_tools.py  
Contains multiple helper functions that are needed to preprocess the input files before simulation: creating valves, changing the assigned demand and other tasks. (Current Working version: v2)  
  
## 4. get_equity.py
Post-processes the EPANET output file and returns the value of the equity metric of choice, the Uniformity Coefficient (UC) or Volumetric Coefficient (VC)  

## 5. Optimizer.py  
Our main optimization script. Optimizes the valve settings for a given network over one 24-hr period: with one optimization run for each hour in the demand pattern. (Current Working Version: 3.0)