# File: data_generator.jl
# Generates and saves detailed sampled data for both "smarter" and "CA" policies,
# creating separate CSV files for each grid cell labeled by `i,j`.

using CSV
using DataFrames
using Random
using Statistics
using Distributions
include("../STPMDP_ORS.jl")
include("grid_plot_constants.jl")  # for constants

# Constants
NUM_SAMPLES = 3  # probs will reset this to 10

# Create an instance of the MDP
myMDP = StrokeMDP()

# Function to generate and save sample data for each grid cell
function generate_and_save_data(grid_size, my_region, output_dir)
    lon_min, lon_max, lat_min, lat_max = my_region
    lon_step = (lon_max - lon_min) / grid_size
    lat_step = (lat_max - lat_min) / grid_size

    for i in 1:grid_size
        sub_lon_min = lon_min + (i - 1) * lon_step
        sub_lon_max = lon_min + i * lon_step
        for j in 1:grid_size
            sub_lat_min = lat_min + (j - 1) * lat_step
            sub_lat_max = lat_min + j * lat_step
            cell = (sub_lon_min, sub_lat_min), (sub_lon_max, sub_lat_max)

            # Create a DataFrame for the current cell
            cell_samples_df = DataFrame(
                sample=[],
                reward_best=[],
                reward_CA=[],
                location_lat=[],
                location_lon=[],
                patient_state=[]
            )

            sample_count = 0
            while sample_count < NUM_SAMPLES
                try
                    # Generate a random sample location and stroke type
                    sampled_location = (
                        rand() * (sub_lat_max - sub_lat_min) + sub_lat_min,
                        rand() * (sub_lon_max - sub_lon_min) + sub_lon_min
                    )
                    sampled_stroke_type = sample_stroke_type(myMDP)
                    sampled_start_state = PatientState(
                        Location("FIELD", sampled_location, -1, FIELD),
                        rand() * 270,
                        UNKNOWN,
                        sampled_stroke_type
                    )

                    # Calculate reward under "smarter" (best) policy
                    recommended_action = best_action(myMDP, sampled_start_state, 2)
                    next_state_distribution_best = transition(myMDP, sampled_start_state, recommended_action)
                    next_state_best = rand(next_state_distribution_best)
                    reward_best = reward(myMDP, sampled_start_state, recommended_action, next_state_best)

                    # Calculate reward under "CA" (status quo) policy
                    nearest_hospital_action_string = current_CApolicy_action(myMDP, sampled_start_state)
                    nearest_hospital_action = string_to_enum(nearest_hospital_action_string)
                    next_state_distribution_CA = transition(myMDP, sampled_start_state, nearest_hospital_action)
                    next_state_CA = rand(next_state_distribution_CA)
                    reward_CA = reward(myMDP, sampled_start_state, nearest_hospital_action, next_state_CA)

                    # Add the sample to the DataFrame
                    push!(cell_samples_df, (
                        sample_count + 1,
                        reward_best,
                        reward_CA,
                        sampled_location[1],
                        sampled_location[2],
                        string(sampled_start_state)
                    ))

                    sample_count += 1  # Increment only on successful routing
                catch e
                    println("Error occurred during routing for cell ($i, $j), retrying: $e")
                end
            end

            # Ensure the output directory exists
            if !isdir(output_dir)
                mkpath(output_dir)
            end

            # Save the results for the current cell to a CSV file
            cell_output_file = joinpath(output_dir, "samples_data_$(i)_$(j).csv")
            CSV.write(cell_output_file, cell_samples_df)
            println("Saved data for cell ($i, $j) to: $cell_output_file")
        end
    end
end

# Generate and save data
println("Generating data...")
generate_and_save_data(GRID_SIZE, MY_REGION, OUTPUT_DIR)
println("Data generation complete!")
