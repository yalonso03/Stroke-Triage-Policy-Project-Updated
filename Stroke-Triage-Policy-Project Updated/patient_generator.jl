#= 
Modeling Stroke Patient Triage and Transport in CA with MDPs
Autumn 2023 CS199/CS195 Project
Emily Molins and Yasmine Alonso

File: patient_generator_CA.jl
--------------------
This code simulates a single stroke patient scenario in California, generating a patient state and comparing the outcomes 
of an MDP-based optimal triage policy with the current state policy by analyzing recommended actions, rewards, and travel times.
=#

using CSV
using DataFrames
using StatsBase
using Random
using Plots
using StatsPlots
using ProgressBars
using Distributions
using HTTP
using JSON
include("STPMDP_ORS.jl")

# Function to generate and display a single patient's information
function generate_and_display_patient(myMDP)
    # Sample a random patient state
    location = sample_location()
    onset_time = rand() * 270  # Random onset time within the 270-minute range
    stroke_type = sample_stroke_type(myMDP)
    patient_state = PatientState(Location("FIELD", location, -1, FIELD), onset_time, UNKNOWN, stroke_type)

    println("Patient Information:")
    println("  Location: $(patient_state.loc.latlon)")
    println("  Stroke Type: $stroke_type")
    println("  Onset Time: $(round(onset_time, digits=2)) minutes")

    # Evaluate MDP-based policy
    recommended_action = best_action(myMDP, patient_state, 2)
    next_state = rand(transition(myMDP, patient_state, recommended_action))
    best_action_reward = reward(myMDP, patient_state, recommended_action, next_state)
    travel_time = calculate_travel_time(patient_state.loc, next_state.loc)

    println("\nMDP-Based Policy:")
    println("  Recommended Action: $(enum_to_string(recommended_action))")
    println("  Reward: $best_action_reward")
    println("  Travel Time: $(round(travel_time, digits=2)) minutes")

    # Evaluate nearest hospital policy
    nearest_hospital_action_string = current_CApolicy_action(myMDP, patient_state)
    nearest_hospital_action = string_to_enum(nearest_hospital_action_string)
    nh_next_state = rand(transition(myMDP, patient_state, nearest_hospital_action))
    nearest_hospital_reward = reward(myMDP, patient_state, nearest_hospital_action, nh_next_state)
    travel_time_nh = calculate_travel_time(patient_state.loc, nh_next_state.loc)

    println("\nNearest Hospital Policy:")
    println("  Recommended Action: $nearest_hospital_action_string")
    println("  Reward: $nearest_hospital_reward")
    println("  Travel Time: $(round(travel_time_nh, digits=2)) minutes")
end

# Generate and display one patient as an example
generate_and_display_patient(myMDP)