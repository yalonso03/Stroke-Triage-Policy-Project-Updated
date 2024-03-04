#=
Modeling Stroke Patient Triage and Transport in CA with MDPs
Winter 2024 CS239
Emily Molins and Yasmine Alonso

File: evaluate_policies.jl
--------------------
Compares our smarter policy with the baseline (CA's current policy) and some heuristic policies. 
=#

include("STPMDP.jl")
include("simulations.jl")

using Plots

N_SIMULATIONS = 1000


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
mutable struct SimulationResultGeneral
    rewards::Vector{Float64}  # 1: reward for best policy, #2: reward for nearest hospital (CA CURRENT), #3 reward for Heuristic #1, #4: reward for Heuristic #2
    travel_times::Vector{Float64}  # 1: travel_time for best policy, #2: travel_time for nearest hospital (CA CURRENT), #3 travel_time for Heuristic #1, #4: travel_time for Heuristic #2
    suggested_actions::Vector{String} 
    start_state::PatientState
end



myMDP = StrokeMDP()

function runsims(myMDP)
    results = [] #Vector{SimulationResultGeneral}()
    iter = ProgressBar(1:N_SIMULATIONS)
    for i in iter
        # Sample the starting state
        sampled_start_state = PatientState(Location("FIELD1", sample_location(), -1, FIELD), rand() * 270, UNKNOWN, sample_stroke_type(myMDP))
        
        # Calculate the reward for the best action according to our SMARTER POLICY
        recommended_action = best_action(myMDP, sampled_start_state, 2)
        action_string_smarter = enum_to_string(recommended_action)
        next_state_distribution = transition(myMDP, sampled_start_state, recommended_action)
        next_state = rand(next_state_distribution)
        best_action_reward = reward(myMDP, sampled_start_state, recommended_action, next_state)
        
        # Calculate the reward for taking the patient to the nearest hospital (CURRENT CA POLICY)
        nearest_hospital_action_string = current_CApolicy_action(myMDP, sampled_start_state)
        nearest_hospital_action = string_to_enum(nearest_hospital_action_string)
        action_string_nh = enum_to_string(nearest_hospital_action)
        nh_next_state_distribution = transition(myMDP, sampled_start_state, nearest_hospital_action)
        nh_next_state = rand(nh_next_state_distribution)
        nearest_hospital_reward = reward(myMDP, sampled_start_state, nearest_hospital_action, nh_next_state)

        # Calculate the reward for routing the patient by a heuristic policy #1 (ROUTE TO NEAREST CSC)
        h1_action_string = heuristic_1_action(myMDP, sampled_start_state)
        h1_action = string_to_enum(h1_action_string)
        h1_next_state_distribution = transition(myMDP, sampled_start_state, h1_action)
        h1_next_state = rand(h1_next_state_distribution)
        h1_reward = reward(myMDP, sampled_start_state, h1_action, h1_next_state)

        # Calculate the reward for routing the patient by a heuristic policy #2 (ROUTE TO NEAREST PSC)
        h2_action_string = heuristic_2_action(myMDP, sampled_start_state)
        h2_action = string_to_enum(h2_action_string)
        h2_next_state_distribution = transition(myMDP, sampled_start_state, h2_action)
        h2_next_state = rand(h2_next_state_distribution)
        h2_reward = reward(myMDP, sampled_start_state, h2_action, h2_next_state)

        # Calculate travel times for smarter policy, nearest hospital, and heuristic policies 1 and 2
        travel_time_smarter = calculate_travel_time(sampled_start_state.loc, next_state.loc)
        travel_time_nh = calculate_travel_time(sampled_start_state.loc, nh_next_state.loc)
        travel_time_h1 = calculate_travel_time(sampled_start_state.loc, h1_next_state.loc)
        travel_time_h2 = calculate_travel_time(sampled_start_state.loc, h2_next_state.loc)

        rewards = [best_action_reward, nearest_hospital_reward, h1_reward, h2_reward]
        travel_times = [travel_time_smarter, travel_time_nh, travel_time_h1, travel_time_h2]
        actions = [action_string_smarter, action_string_nh, h1_action_string, h2_action_string]

        # Store the results
        push!(results, SimulationResultGeneral(rewards, travel_times, actions, sampled_start_state))
    end
    return results
end


# # lst is a vector of vector of rewards, or vector of vector travel times, for example
# # lst is length 4 (one vector for smarter, CA, h1, h2)
# # reward_or_times is either "r" or "t" to represent what we are plotting
# function make_histogram(lst_of_lsts, reward_or_times, policy_type)
#     # maximum value 
#     max_val = max(maximum(lst_of_lsts[1]), maximum(lst_of_lsts[2]), maximum(lst_of_lsts[3]), maximum(lst_of_lsts[4]))
#     min_val = min(minimum(lst_of_lsts[1]), minimum(lst_of_lsts[2]), minimum(lst_of_lsts[3]), minimum(lst_of_lsts[4]))
    
#     bin_range = range(min_val, max_val, length=21) #20 bins in this range

#     # [1] is the smarter policy
#     if reward_or_times == "r"
#         xlabel = "Total Accumulated Reward under " * policy_type * "Policy"
#     end

#     if policy_type == "Smarter"
#         i = 1
#     elseif policy_type == "CA"
#         i = 2
#     elseif policy_type == "Heuristic 1"
#         i = 3
#     else
#         i = 4
#     end

#     histogram(lst_of_lsts[i], bins=bin_range, xlabel=xlabel, ylabel="Frequency", title="Total Accumulated Reward under " * policy_type * " with 1000 Simulations")
    
# end


results = runsims(myMDP)

function analyze_results(results, show_plots)
    smarter_rewards = []
    nh_rewards = []
    h1_rewards = []
    h2_rewards = []

    smarter_times = []
    nh_times = []
    h1_times = []
    h2_times = []

    smarter_actions = []
    nh_actions = []
    h1_actions = []
    h2_actions = []

    for result in results
        push!(smarter_rewards, result.rewards[1])
        push!(nh_rewards, result.rewards[2])
        push!(h1_rewards, result.rewards[3])
        push!(h2_rewards, result.rewards[4])

        push!(smarter_times, result.travel_times[1])
        push!(nh_times, result.travel_times[2])
        push!(h1_times, result.travel_times[3])
        push!(h2_times, result.travel_times[4])

        push!(smarter_actions, result.suggested_actions[1])
        push!(nh_actions, result.suggested_actions[2])
        push!(h1_actions, result.suggested_actions[3])
        push!(h2_actions, result.suggested_actions[4])
    end

    # Calculate average reward and average time 
    avg_smarter_rewards = mean(smarter_rewards)
    avg_nh_rewards = mean(nh_rewards)
    avg_h1_rewards = mean(h1_rewards)
    avg_h2_rewards = mean(h2_rewards)

    avg_smarter_times = mean(smarter_times)
    avg_nh_times = mean(nh_times)
    avg_h1_times = mean(h1_times)
    avg_h2_times = mean(h2_times)

    # Calculate median reward and median times
    med_smarter_rewards = median(smarter_rewards)
    med_nh_rewards = median(nh_rewards)
    med_h1_rewards = median(h1_rewards)
    med_h2_rewards = median(h2_rewards)

    med_smarter_times = median(smarter_times)
    med_nh_times = median(nh_times)
    med_h1_times = median(h1_times)
    med_h2_times = median(h2_times)

   
    std_smarter_rewards = std(smarter_rewards)
    std_nh_rewards = std(nh_rewards)
    std_h1_rewards = std(h1_rewards)
    std_h2_rewards = std(h2_rewards)

    std_smarter_times = std(smarter_times)
    std_nh_times = std(nh_times)
    std_h1_times = std(h1_times)
    std_h2_times = std(h2_times)

    println("Average reward accrued through our policy: ", avg_smarter_rewards)
    println("Average reward accrued through CA's current policy: ", avg_nh_rewards)
    println("Average reward accrued through routing to the nearest CSC (Heuristic 1): ", avg_h1_rewards)
    println("Average reward accrued through routing to the nearest CSC/PSC (Heuristic 2): ", avg_h2_rewards)

    println()

    println("Median reward accrued through our policy: ", med_smarter_rewards)
    println("Median reward accrued through CA's current policy: ", med_nh_rewards)
    println("Median reward accrued through routing to the nearest CSC (Heuristic 1): ", med_h1_rewards)
    println("Median reward accrued through routing to the nearest CSC/PSC (Heuristic 2): ", med_h2_rewards)

    if show_plots == true
        lst_of_lsts = [smarter_rewards, nh_rewards, h1_rewards, h2_rewards]
        max_val = max(maximum(lst_of_lsts[1]), maximum(lst_of_lsts[2]), maximum(lst_of_lsts[3]), maximum(lst_of_lsts[4]))
        min_val = min(minimum(lst_of_lsts[1]), minimum(lst_of_lsts[2]), minimum(lst_of_lsts[3]), minimum(lst_of_lsts[4]))
        
        bin_range = range(min_val, max_val, length=21) #20 bins in this range
        p1 = histogram(lst_of_lsts[1], bins=bin_range, xlabel="Reward under Smarter Policy", ylabel="Frequency", color=:green) #, title="Total Accumulated Reward under Smarter Policy with 1000 Simulations")
        vline!(p1, [avg_smarter_rewards], linestyle=:dash, color=:red, label="Avg Reward")
        p2 = histogram(lst_of_lsts[2], bins=bin_range, xlabel="Reward under CA Current Policy", ylabel="Frequency", color=:red) #, title="Total Accumulated Reward under CA Current Policy with 1000 Simulations")
        vline!(p2, [avg_smarter_rewards], linestyle=:dash, color=:red, label="Avg Reward")
        vline!(p2, [avg_nh_rewards], linestyle=:dash, color=:black, label="Avg Reward")
        p3 = histogram(lst_of_lsts[3], bins=bin_range, xlabel="Reward under Nearest CSC (H1) Policy", ylabel="Frequency", color=:orange) #, title="Total Accumulated Reward under Heuristic 1 Policy with 1000 Simulations")
        vline!(p3, [avg_smarter_rewards], linestyle=:dash, color=:red, label="Avg Reward")
        vline!(p3, [avg_h1_rewards], linestyle=:dash, color=:black, label="Avg Reward")
        p4 = histogram(lst_of_lsts[4], bins=bin_range, xlabel="Reward under Nearest CSC/PSC (H2) Policy", ylabel="Frequency", color = :blue) #, title="Total Accumulated Reward under Heuristic 2 Policy with 1000 Simulations")
        vline!(p4, [avg_smarter_rewards], linestyle=:dash, color=:red, label="Avg Reward")
        vline!(p4, [avg_h2_rewards], linestyle=:dash, color=:black, label="Avg Reward")
        plot(p1, p2, p3, p4, layout=(2, 2), legend=false)
        
    end
end

analyze_results(results, true)

