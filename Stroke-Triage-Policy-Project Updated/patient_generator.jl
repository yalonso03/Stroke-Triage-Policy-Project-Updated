#=
Modeling Stroke Patient Triage and Transport in CA with MDPs
Autumn 2023 CS199/CS195 Project
Emily Molins and Yasmine Alonso

File: simulations.jl
--------------------
This file runs many simulations of a random patient (with random field location, random stroke type), and produces the 
recommended actions to take (via our policy, and California's current policy). Uses STPMDP.jl to produce the MDP.
=#

using CSV
using DataFrames
using DelimitedFiles
using D3Trees
using DiscreteValueIteration
using Graphs
using LaTeXStrings
using LinearAlgebra
using LocalApproximationValueIteration
using MCTS
using POMDPModels
using POMDPModelTools
using POMDPTools
using POMDPs
using Parameters
using Plots
using Printf
using PyCall
using QuickPOMDPs
using QMDP
using Random
using RollingFunctions
using SpecialFunctions
using Statistics
using TabularTDLearning
using LinearAlgebra
using StatsBase
using GR
using StatsPlots
using Distributions
using ProgressBars

include("STPMDP.jl")

N_SIMULATIONS = 2000


# Ok i think googleAPI functions failing because above sample_location may be coming up with points in the water lol and then u cant
# get driving directions. for now 30 random points 
function sample_location()
    # Define the filename for the CSV file
    csv_filename = "PatientPoints/sample_points.csv"

    df = CSV.File(csv_filename) |> DataFrame

    # Extract latitude and longitude columns from the DataFrame
    latitudes = df[:, "Latitude"]
    longitudes = df[:, "Longitude"]

    # Create a list of tuples from the latitude and longitude columns
    location_list = [(latitudes[i], longitudes[i]) for i in 1:size(df, 1)]

    # Generate a random index to select a point from the list
    random_index = rand(1:length(location_list))

    # Return the randomly selected (latitude, longitude) point
    return location_list[random_index]
end

# Define a structure to hold the results
mutable struct SimResult
    best_action_reward::Float64
    nearest_hospital_reward::Float64
    #nearest_hospital_sp::PatientState
    start_state::PatientState
end


# Function to run a single simulation and print the details
function run_single_simulation(myMDP)
    # Sample the starting state
    sampled_location = sample_location()
    sampled_stroke_type = sample_stroke_type(myMDP)
    sampled_start_state = PatientState(Location("FIELD", sampled_location, 0, FIELD), rand() * 120, UNKNOWN, sampled_stroke_type)

    println("Sampled Location: ", sampled_location)
    println("Sampled Stroke Type: ", sampled_stroke_type)
    
    # Get and print the available actions for the sampled_start_state
    available_actions = POMDPs.actions(myMDP, sampled_start_state)
    println("Available Actions: ", available_actions)
    
    # Calculate the reward for the best action
    recommended_action = best_action(myMDP, sampled_start_state, 1)
    next_state_distribution = transition(myMDP, sampled_start_state, recommended_action)
    next_state = rand(next_state_distribution)
    best_action_reward = reward(myMDP, sampled_start_state, recommended_action, next_state)
    
    println("Recommended Action: ", recommended_action)
    println("Resulting State: ", next_state)
    println("Reward for Best Action: ", best_action_reward)
    
    # Calculate the reward for taking the patient to the nearest hospital (Current CA policy)
    nearest_hospital_action_string = current_CApolicy_action(myMDP, sampled_start_state)
    nearest_hospital_action = string_to_enum(nearest_hospital_action_string)
    nh_next_state_distribution = transition(myMDP, sampled_start_state, nearest_hospital_action)
    nh_next_state = rand(nh_next_state_distribution)
    nearest_hospital_reward = reward(myMDP, sampled_start_state, nearest_hospital_action, nh_next_state)
    
    println("Nearest Hospital Action: ", nearest_hospital_action)
    println("Nearest Hospital Next State: ", nh_next_state)
    println("Reward for Nearest Hospital: ", nearest_hospital_reward)

    # Return the results as a dictionary or custom struct
    return Dict(
        "sampled_location" => sampled_location,
        "sampled_stroke_type" => sampled_stroke_type,
        "recommended_action" => recommended_action,
        "next_state" => next_state,
        "best_action_reward" => best_action_reward,
        "nearest_hospital_action" => nearest_hospital_action,
        "nearest_hospital_next_state" => nh_next_state,
        "nearest_hospital_reward" => nearest_hospital_reward
    )
end

# Run the single simulation
myMDP = StrokeMDP()
simulation_result = run_single_simulation(myMDP)

# Print the simulation result
println("Simulation Result: ", simulation_result)
