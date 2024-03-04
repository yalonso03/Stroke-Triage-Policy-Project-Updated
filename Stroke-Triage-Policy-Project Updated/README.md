# Stroke-Triage-Policy-Project
Our project continuing work in Winter 2024 for CS239 on the routing of stroke patients to nearest hospitals to optimize probability of a good patient outcome.

An overview of the repo:
**STPMDP.jl** contains the implementation of the MDP. This is the main file of interest.

**Figures** is where we store all figures and plots produced in our simulations.

**SimResults** directory contains results of simulations we have run as CSV files to save for later use.

**action_enum.jl** is a file that allows for the quick definition of our Action enum type for use in STPMDP.jl. It reads from a CSV file that lists all hospitals in the region.

**grid_plot.jl** produces plots that aim to show the probability of a good outcome for a patient in a gridded-up version of the focus area (for now, SF bay area).

**hospital_list_CA.csv** is a CSV file containing all of our documented hospitals and their corresponding information (i.e. location, label as a Clinic/PSC/CSC) for the Bay Area.

**map_plot.jl** produces plots of the SF Bay area that show simulation results as points on the map, color coded by the probability of a good outcome of that patient point. 

**patient_generator.jl** runs a single simulation of a patient -- **TODO ask Emily about purpose of this file exactly**

**sample_points.csv** is a CSV file containing approximately 5000 randomly sampled locations, by population density, masked to being on land.

**simulations.jl** runs N_SIMULATIONS number of patient simulations and also calculates some basic patient outcome statistics, and produces some plots on those statistics.   
