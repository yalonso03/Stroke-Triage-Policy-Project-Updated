# File: grid_plot.jl by Emily Molins and Yasmine Alonso for AA229
# -----------------------------------------------------------
# The grid_plot.jl file utilizes the GMT library in order to produce plots 
# that aim to show the probability of a good outcome for a patient in a 
# gridded-up version of the focus area (SF bay area). To do so, we grid
# up the bay area into a grid_size by grid_size grid (typically 200x200)
# and sample 10 patients per square and their stroke types (from population statistics) 
# from random locations in each square in the grid. We take an average probability
# of good outcome in each square from those 10 sampled patients, color-coding each
# grid square by this average probability. This file then produces 2 plots: one 
# that routes patients via the Status Quo Policy CA currently uses, and one via
# our "smarter" policy that is the result of solving the MDP defined in STPMDP.jl
# with forward search.

using GMT
using CSV
using DataFrames
using DelimitedFiles
using LinearAlgebra
using POMDPModels
using POMDPs
using Random
using Statistics
using Distributions

include("../STPMDP_original.jl")

myMDP = StrokeMDP()

# The sample_and_average function is called for each square in the overall grid, and produces num_samples (usually 10)
# patients which are randomly produced (and where relevant, weighted by population stats, such as when we sample stroke type).
# It takes in a grid_cell, which defines the coordinates of the vertices of that cell, an instance of our stroke MDP, a number
# of simulated patients we'd like to produce (num_samples), and an option, where option is either "smarter" or "CA" depending
# on which way we'd like to determine the routing action to take for each simulated patient.
# Out: an average "probability of a good outcome" for all of the num_sampled patients after taking the routing action decided by
# either the current CA status quo policy or our smarter policy.
function sample_and_average(grid_cell, MDP, num_samples, option)
    (sample_lon_min, sample_lat_min), (sample_lon_max, sample_lat_max) = grid_cell
    avg_prob = 0.0

    for _ in 1:num_samples
        sampled_location = (rand() * (sample_lat_max - sample_lat_min) + sample_lat_min, rand() * (sample_lon_max - sample_lon_min) + sample_lon_min)
        #println("sampled location is:", sampled_location)
        # Simulate a patient at the sampled location using MDP
        sampled_stroke_type = sample_stroke_type(MDP)
        sampled_start_state = PatientState(Location("FIELD", sampled_location, -1, FIELD), rand() * 270, UNKNOWN, sampled_stroke_type)

        if option=="smarter"
        # Our smarter policy simulation
            sampled_start_state_smarter = PatientState(Location("FIELD", sampled_location, -1, FIELD), rand() * 270, UNKNOWN, sampled_stroke_type)
            recommended_action = best_action(MDP, sampled_start_state_smarter, 2)
            next_state_distribution = transition(myMDP, sampled_start_state_smarter, recommended_action)
            next_state = rand(next_state_distribution)
            best_action_reward = reward(myMDP, sampled_start_state, recommended_action, next_state)
            avg_prob += best_action_reward
        else
            # CA policy simulation
            nearest_hospital_action_string = current_CApolicy_action(MDP, sampled_start_state)
            nearest_hospital_action = string_to_enum(nearest_hospital_action_string)
            nh_next_state_distribution = transition(MDP, sampled_start_state, nearest_hospital_action)
            nh_next_state = rand(nh_next_state_distribution)
            nearest_hospital_reward = reward(MDP, sampled_start_state, nearest_hospital_action, nh_next_state)
            avg_prob += nearest_hospital_reward
        end
    end
    return avg_prob / num_samples
end


# The make_plot_with_grid function produces a plot showing the average probability of a good outcome under either option
# "smarter" or "CA"
function make_plot_with_grid(option)
    # Define the latitude and longitude of Hayward, CA
    hayward_lat = 37.6688
    hayward_lon = -122.0808
    zoom_level = 0.5

    # Calculate the region coordinates
    my_region = [hayward_lon - zoom_level, hayward_lon + zoom_level, hayward_lat - zoom_level, hayward_lat + zoom_level]
    lon_min, lon_max, lat_min, lat_max = my_region

    #! DIMENSIONS OF THE GRID
    grid_size = 3

    lon_step = (lon_max - lon_min) / grid_size
    lat_step = (lat_max - lat_min) / grid_size

    X = [lon_min]
    Y = [lat_min]

    colors = zeros(grid_size, grid_size)
    probs = []
    for i in 1:grid_size
        sub_lon_min = lon_min + i * lon_step
        sub_lon_max = lon_min + (i + 1) * lon_step
        for j in 1:grid_size 
            sub_lat_min = lat_min + j * lat_step
            sub_lat_max = lat_min + (j + 1) * lat_step
            cell = (sub_lon_min, sub_lat_min), (sub_lon_max, sub_lat_max)
            prob = sample_and_average(cell, myMDP, 10, option)
            push!(probs, prob)
            colors[i, j] = prob
            # Check i == 1 to ensure that we only add to Y grid_size times (not grid_size * grid_size times)
            if i == 1 && j != grid_size
                push!(Y, sub_lat_min)
            end
        end
        if i != grid_size
            push!(X, sub_lon_min)
        end
    end


    # Create inverted landmask grid to not color any squares on water (swap the values for land and water)
    landmask = grdlandmask(region=my_region, spacing=(lon_step, lat_step), resolution="f", maskvalues=(NaN, 1))


    #Apply mask to grid data, masking out (set to Nan) if the grid square is in the water
    for i in 1:grid_size
        for j in 1:grid_size
            if isnan(landmask[i, j]) 
                colors[i, j] =  NaN
            end
        end
    end

    # Define the title based on the option
    title = option == "smarter" ? "Probability of Good Outcome by Region under Optimal Policy" : "Probability of Good Outcome by Region under Status Quo Policy"

    # Plotting
    cpt = makecpt(color=:hot, range=(0.1,0.4,.001))

    #println(my_region)
    pcolor(X, Y, colors, cmap=cpt, proj="merc", title=title)

    path = "Stroke-Triage-Policy-Project Updated/Figures" * option * "_hotspot_gridplot.pdf"
    # colorbar!(cmap=cpt, title="Probability of Good Outcome", projection=:mercator)


    #coast!(region=my_region, title=title, savefig=path, projection=:mercator, show=true) # make the map
    
    coast!(region=my_region, savefig=path, show=true, proj="merc")

end

println("producing plot of p_good_outcome simulations with our smarter policy")
make_plot_with_grid("smarter")
println("producing plot for CA's current policy")
make_plot_with_grid("CA")