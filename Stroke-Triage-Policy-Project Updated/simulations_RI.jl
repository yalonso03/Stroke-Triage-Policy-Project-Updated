#=
Modeling Stroke Patient Triage and Transport in CA with MDPs
Autumn 2023 CS199/CS195 Project
Emily Molins and Yasmine Alonso

File: simulations_RI.jl
--------------------
This file runs many simulations of a random patient (with random field location, random stroke type), and produces the 
recommended actions to take (via our policy, and California's current policy). Uses STPMDP.jl to produce the MDP.

This is for the Rhode Island simulations, as their status quo policy is different (only routes to nearest CSC)
=#

using CSV
using DataFrames
using StatsBase
using Random
using Plots
using StatsPlots
using ProgressBars
using Distributions

Random.seed!(1) 

include("STPMDP_ORS.jl")

N_SIMULATIONS = 3


# Ok i think googleAPI functions failing because above sample_location may be coming up with points in the water lol and then u cant
# get driving directions. for now 30 random points 
function sample_location()
    # Define the filename for the CSV file
    csv_filename = "Stroke-Triage-Policy-Project Updated/PatientPoints/points_on_land_RI.csv"

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
mutable struct SimulationResult
    best_action_reward::Float64
    nearest_hospital_reward::Float64
    travel_time::Float64
    travel_time_nh::Float64
    start_state::PatientState
    recommended_action::String  # Added field to store recommended action
    nearest_hospital_action::String  # Added field to store nearest hospital action
end

# Initialize an array to store the results of each simulation

myMDP = StrokeMDP()

function run_simulations(myMDP)
    results = Vector{SimulationResult}()
    iter = ProgressBar(1:N_SIMULATIONS)
    for i in iter
        # Sample the starting state
        sampled_start_state = PatientState(Location("FIELD1", sample_location(), -1, FIELD), rand() * 270, UNKNOWN, sample_stroke_type(myMDP))
        
        # Calculate the reward for the best action
        recommended_action = best_action(myMDP, sampled_start_state, 2)
        action_string = enum_to_string(recommended_action)
        next_state_distribution = transition(myMDP, sampled_start_state, recommended_action)
        next_state = rand(next_state_distribution)
        best_action_reward = reward(myMDP, sampled_start_state, recommended_action, next_state)
        
        # Calculate the reward for taking the patient to the single CSC (Current RI policy)
        CSC_action_string = heuristic_1_action(myMDP, sampled_start_state)
        CSC_action = string_to_enum(CSC_action_string)
        action_string_CSC = enum_to_string(CSC_action)
        CSC_next_state_distribution = transition(myMDP, sampled_start_state, CSC_action)
        CSC_next_state = rand(CSC_next_state_distribution)
        CSC_reward = reward(myMDP, sampled_start_state, CSC_action, CSC_next_state)
        
        # Step 1: Calculate travel times for the first action
        travel_time = calculate_travel_time(sampled_start_state.loc, next_state.loc)
        travel_time_CSC = calculate_travel_time(sampled_start_state.loc, nh_next_state.loc)


        # Store the results
        push!(results, SimulationResult(best_action_reward, CSC_reward, travel_time, travel_time_CSC, 
        sampled_start_state, action_string, action_string_CSC))
    end
    return results
end

# Initialize the arrays to store the rewards
best_action_rewards = Float64[]
CSC_rewards = Float64[]



# Results stores our simulation results
results = run_simulations(myMDP)

# Extract rewards from the results
for result in results
    push!(best_action_rewards, result.best_action_reward)
    push!(CSC_rewards, result.CSC_reward)
end

outfile = open("Stroke-Triage-Policy-Project Updated/SimResults/TEST2.csv", "w")
# Put header
# Note that nearest_hospital_reward column is actually reward routing to only CSC
println(outfile, "best_action_reward%nearest_hospital_reward%start_state_latlon")
# Using percent sign as delimeter so we can do file reading
for result in results
    print(outfile, string(result.best_action_reward)*'%')
    print(outfile, string(result.nearest_hospital_reward)*'%')
    print(outfile, string(result.start_state.loc.latlon))
    print(outfile, '\n')
end
close(outfile)

# Analyze the results
# Calculate the average reward for each strategy
avg_best_action_reward = mean(best_action_rewards)
avg_nearest_hospital_reward = mean(nearest_hospital_rewards)

println("Average reward for best action: ", avg_best_action_reward)
println("Average reward for RI current policy: ", avg_nearest_hospital_reward)

# Compute the median of the rewards
median_best_action_reward = median(best_action_rewards)
median_nearest_hospital_reward = median(nearest_hospital_rewards)

println("Median reward for best action: ", median_best_action_reward)
println("Median reward for RI current policy: ", median_nearest_hospital_reward)

# Compute the standard deviation of the rewards
std_best_action_reward = std(best_action_rewards)
std_nearest_hospital_reward = std(nearest_hospital_rewards)

println("Standard deviation of reward for best action: ", std_best_action_reward)
println("Standard deviation of reward for RI current policy: ", std_nearest_hospital_reward)

gr() 

# Convert the results to a format suitable for plotting
best_action_rewards = [r.best_action_reward for r in results]
nearest_hospital_rewards = [r.nearest_hospital_reward for r in results]
data = [best_action_rewards nearest_hospital_rewards]

# Initialize the variables outside the loop
global max_increase = -Inf
global max_increase_simulation = nothing

for i in 1:length(results)
    # Explicitly declare the variables as global inside the loop
    global max_increase, max_increase_simulation

    difference = results[i].best_action_reward - results[i].nearest_hospital_reward

    if difference > max_increase
        max_increase = difference
        max_increase_simulation = results[i]
    end
end

if max_increase_simulation !== nothing
    println("----------------------------------------------------------------------")
    println("Maximum increase in reward: ", max_increase)
    println("Occurred in simulation with start state: ", max_increase_simulation.start_state)
    println("Probability of good outcome with best action: ", max_increase_simulation.best_action_reward)
    println("Probability of good outcome with RI current policy: ", max_increase_simulation.nearest_hospital_reward)
else
    println("No simulation with increase found.")
end

# Plot the results using a box plot
StatsPlots.boxplot(["Best Action" "RI current policy"], data, title="Comparison of Rewards", ylabel="Probability of Good Outcome", legend=false)

# Extract travel times
best_action_travel_times = [r.travel_time for r in results]
nearest_hospital_travel_times = [r.travel_time_nh for r in results]

# Calculate statistics
avg_best_action_travel_time = mean(best_action_travel_times)
avg_nearest_hospital_travel_time = mean(nearest_hospital_travel_times)
median_best_action_travel_time = median(best_action_travel_times)
median_nearest_hospital_travel_time = median(nearest_hospital_travel_times)
std_best_action_travel_time = std(best_action_travel_times)
std_nearest_hospital_travel_time = std(nearest_hospital_travel_times)

# Printing out the statistics
println("----------------------------------------------------------------------")
println("Average Travel Time (Best Action): ", avg_best_action_travel_time, " minutes")
println("Average Travel Time (RI current policy): ", avg_nearest_hospital_travel_time, " minutes")
println("Median Travel Time (Best Action): ", median_best_action_travel_time, " minutes")
println("Median Travel Time (RI current policy): ", median_nearest_hospital_travel_time, " minutes")
println("Standard Deviation of Travel Time (Best Action): ", std_best_action_travel_time, " minutes")
println("Standard Deviation of Travel Time (RI current policy): ", std_nearest_hospital_travel_time, " minutes")


# Visualizing Hospitals Routed To
recommended_action_counts = Dict()
nearest_hospital_action_counts = Dict()

for result in results
    recommended_action_counts[result.recommended_action] = get(recommended_action_counts, result.recommended_action, 0) + 1
    nearest_hospital_action_counts[result.nearest_hospital_action] = get(nearest_hospital_action_counts, result.nearest_hospital_action, 0) + 1
end

# Create a combined set of all unique actions from both policies
all_actions_set = union(keys(recommended_action_counts), keys(nearest_hospital_action_counts))

# Convert the set to a sorted list for consistent ordering
all_actions = sort(collect(all_actions_set))

# Initialize vectors to store counts for each action for both policies
recommended_action_values = []
nearest_hospital_action_values = []

# Populate the count vectors, using 0 for actions that don't appear in a policy
for action in all_actions
    push!(recommended_action_values, get(recommended_action_counts, action, 0))
    push!(nearest_hospital_action_values, get(nearest_hospital_action_counts, action, 0))
end

# Number of unique actions for the recommended policy
num_unique_actions_recommended = length(keys(recommended_action_counts))

# Number of unique actions for the RI current policy policy
num_unique_actions_nearest_hospital = length(keys(nearest_hospital_action_counts))
println("----------------------------------------------------------------------")
println("Number of hospitals routed to (Recommended Policy): ", num_unique_actions_recommended)
println("Number of hospitals routed to (RI current policy): ", num_unique_actions_nearest_hospital)


# Calculate basic statistics for recommended actions
mean_recommended = mean(values(recommended_action_counts))
median_recommended = median(values(recommended_action_counts))
std_recommended = std(values(recommended_action_counts))

# Calculate basic statistics for RI current policy actions
mean_nearest_hospital = mean(values(nearest_hospital_action_counts))
median_nearest_hospital = median(values(nearest_hospital_action_counts))
std_nearest_hospital = std(values(nearest_hospital_action_counts))

println("Recommended Action Statistics:")
println("Mean: ", mean_recommended, ", Median: ", median_recommended, ", Standard Deviation: ", std_recommended)

println("RI current policy Action Statistics:")
println("Mean: ", mean_nearest_hospital, ", Median: ", median_nearest_hospital, ", Standard Deviation: ", std_nearest_hospital)