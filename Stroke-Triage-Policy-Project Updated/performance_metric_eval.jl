#= 
Modeling Stroke Patient Triage and Transport in CA with MDPs
Autumn 2023 CS199/CS195 Project
Emily Molins and Yasmine Alonso

File: sims_performance_metrics.jl
--------------------

Summary: This script models stroke patient triage and transport using Markov Decision Processes (MDPs) in 
California. It simulates results across performance metrics to test validity of results.
=#

# Required Libraries
using CSV
using DataFrames
using StatsBase
using Random
using Plots
using StatsPlots
using ProgressBars
using Distributions

# Set Random Seed
Random.seed!(1)

# Include External Dependencies
include("STPMDP_original.jl")

# Constants
const N_SIMULATIONS = 100
const PERFORMANCE_METRIC_START = 60
const PERFORMANCE_METRIC_END = 120
const PERFORMANCE_METRIC_STEP = 1

# Helper Functions

"""
Samples a random location from a CSV file of points in California.
Returns a tuple (latitude, longitude).
"""
function sample_location()
    csv_filename = "sample_points_CA.csv"
    df = CSV.File(csv_filename) |> DataFrame
    location_list = [(df[i, "Latitude"], df[i, "Longitude"]) for i in 1:size(df, 1)]
    return location_list[rand(1:length(location_list))]
end

"""
Samples a random stroke type based on probabilities defined in the MDP.
"""
function sample_stroke_type(MDP)
    probabilities = [MDP.p_LVO, MDP.p_nLVO, MDP.p_Hemorrhagic, MDP.p_Mimic]
    stroke_types = [LVO, NLVO, HEMORRHAGIC, MIMIC]
    return rand(SparseCat(stroke_types, probabilities))
end

"""
Defines a structure to store simulation results.
"""
mutable struct SimulationResult
    best_action_reward::Float64
    nearest_hospital_reward::Float64
    travel_time::Float64
    travel_time_nh::Float64
    start_state::PatientState
    recommended_action::String
    nearest_hospital_action::String
end

"""
Runs simulations for a given MDP and returns a vector of results.
"""
function run_simulations(myMDP)
    results = Vector{SimulationResult}()
    iter = ProgressBar(1:N_SIMULATIONS)
    
    for i in iter
        sampled_start_state = PatientState(Location("FIELD1", sample_location(), -1, FIELD), rand() * 270, UNKNOWN, sample_stroke_type(myMDP))

        # Best action simulation
        recommended_action = best_action(myMDP, sampled_start_state, 2)
        action_string = enum_to_string(recommended_action)
        next_state_distribution = transition(myMDP, sampled_start_state, recommended_action)
        next_state = rand(next_state_distribution)
        best_action_reward = reward(myMDP, sampled_start_state, recommended_action, next_state)

        # Nearest hospital simulation
        nearest_hospital_action_string = current_CApolicy_action(myMDP, sampled_start_state)
        nearest_hospital_action = string_to_enum(nearest_hospital_action_string)
        nh_next_state_distribution = transition(myMDP, sampled_start_state, nearest_hospital_action)
        nh_next_state = rand(nh_next_state_distribution)
        nearest_hospital_reward = reward(myMDP, sampled_start_state, nearest_hospital_action, nh_next_state)

        # Travel times
        travel_time = calculate_travel_time(sampled_start_state.loc, next_state.loc)
        travel_time_nh = calculate_travel_time(sampled_start_state.loc, nh_next_state.loc)

        push!(results, SimulationResult(best_action_reward, nearest_hospital_reward, travel_time, travel_time_nh, sampled_start_state, action_string, nearest_hospital_action_string))
    end

    return results
end

"""
Updates performance metrics for all locations in the MDP.
"""
function update_performance_metrics(locations, new_metric)
    for loc in locations
        if loc.type != FIELD
            loc.performance_metric = new_metric
        end
    end
end

"""
Saves simulation results to a CSV file.
"""
function save_results_to_csv(results, metric)
    directory = "SimResults"
    filename = joinpath(directory, "simulation_results_metric_$(metric).csv")

    if !isdir(directory)
        mkdir(directory)
    end

    open(filename, "w") do file
        println(file, "SimulationIndex,BestActionReward,NearestHospitalReward,TravelTime,TravelTimeNH,StartLat,StartLon,RecommendedAction,NearestHospitalAction")
        for (index, result) in enumerate(results)
            println(file, "$index,$(result.best_action_reward),$(result.nearest_hospital_reward),$(result.travel_time),$(result.travel_time_nh),$(result.start_state.loc.latlon[1]),$(result.start_state.loc.latlon[2]),$(result.recommended_action),$(result.nearest_hospital_action)")
        end
    end

    println("Saved results to $filename")
end

"""
Analyzes and prints key statistics from simulation results.
"""
function analyze_results(results)
    avg_best_action_reward = mean([result.best_action_reward for result in results])
    avg_nearest_hospital_reward = mean([result.nearest_hospital_reward for result in results])
    avg_travel_time = mean([result.travel_time for result in results])
    avg_travel_time_nh = mean([result.travel_time_nh for result in results])

    println("Average Best Action Reward: $avg_best_action_reward")
    println("Average Nearest Hospital Reward: $avg_nearest_hospital_reward")
    println("Average Travel Time (Best Action): $avg_travel_time")
    println("Average Travel Time (Nearest Hospital): $avg_travel_time_nh")
end

"""
Runs simulations across a range of performance metrics and plots the results.
"""
function run_simulations_across_metrics()
    metrics = PERFORMANCE_METRIC_START:PERFORMANCE_METRIC_STEP:PERFORMANCE_METRIC_END
    avg_best_action_rewards = []
    avg_nearest_hospital_rewards = []
    avg_travel_times = []
    avg_travel_times_nh = []

    for metric in metrics
        println("Running simulations for performance metric: ", metric)

        myMDP = StrokeMDP()
        update_performance_metrics(myMDP.locations, metric)

        results = run_simulations(myMDP)
        push!(avg_best_action_rewards, mean([result.best_action_reward for result in results]))
        push!(avg_nearest_hospital_rewards, mean([result.nearest_hospital_reward for result in results]))
        push!(avg_travel_times, mean([result.travel_time for result in results]))
        push!(avg_travel_times_nh, mean([result.travel_time_nh for result in results]))
    end

    p = plot(metrics, avg_best_action_rewards, label="Average Best Action Reward", title="Reward vs Performance Metric", xlabel="Performance Metric", ylabel="Average Reward", legend=:topright)
    plot!(p, metrics, avg_nearest_hospital_rewards, label="Average Nearest Hospital Reward")
    savefig(p, "rewards_vs_metric.png")
end

# Main Execution
run_simulations_across_metrics()
