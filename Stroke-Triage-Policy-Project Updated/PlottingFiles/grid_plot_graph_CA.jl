# File: graph_generator.jl
# Creates plots for both "CA" and "smarter" policies using sampled data saved in per-cell CSV files.

using CSV
using DataFrames
using GMT

# Define constants
const INPUT_DIR = "Stroke-Triage-Policy-Project Updated/PlottingFiles/grid_plot_csvs_CA"

# Function to compute average probabilities for plotting
function compute_probabilities(grid_size, input_dir)
    # Initialize the grid with zeros
    probabilities_best = zeros(grid_size, grid_size)
    probabilities_ca = zeros(grid_size, grid_size)
    sample_counts = zeros(Int, grid_size, grid_size)

    # Iterate over all grid cells
    for i in 1:grid_size
        for j in 1:grid_size
            cell_file = joinpath(input_dir, "samples_data_$(i)_$(j).csv")
            if isfile(cell_file)
                # Check if the first line reads "error"
                first_line = readlines(cell_file)[1]
                if startswith(first_line, "error")
                    println("Cell ($i, $j) contains an error, skipping.")
                    probabilities_best[i, j] = NaN
                    probabilities_ca[i, j] = NaN
                    continue
                end

                # Read and process the CSV if no error is found
                cell_samples_df = CSV.read(cell_file, DataFrame)
                for row in eachrow(cell_samples_df)
                    probabilities_best[i, j] += row.reward_best
                    probabilities_ca[i, j] += row.reward_CA
                    sample_counts[i, j] += 1
                end
            else
                println("File for cell ($i, $j) does not exist, skipping.")
                probabilities_best[i, j] = NaN
                probabilities_ca[i, j] = NaN
            end
        end
    end

    # Compute averages
    for i in 1:grid_size
        for j in 1:grid_size
            if sample_counts[i, j] > 0
                probabilities_best[i, j] /= sample_counts[i, j]
                probabilities_ca[i, j] /= sample_counts[i, j]
            else
                probabilities_best[i, j] = NaN  # No samples for this cell
                probabilities_ca[i, j] = NaN
            end
        end
    end

    return probabilities_best, probabilities_ca
end


# Function to create and save the plots
function make_plot_with_grid(grid_size, my_region, probabilities, option)
    lon_min, lon_max, lat_min, lat_max = my_region
    lon_step = (lon_max - lon_min) / grid_size
    lat_step = (lat_max - lat_min) / grid_size

    # Generate X and Y axis points
    X = range(lon_min, stop=lon_max, length=grid_size + 1)
    Y = range(lat_min, stop=lat_max, length=grid_size + 1)

    # Create color map
    cpt = makecpt(color=:hot, range=(0.1, 0.4, 0.001))

    # Plot grid with probabilities
    pcolor(X, Y, probabilities, cmap=cpt, proj="merc", title=option == "smarter" ? "Optimal Policy" : "Status Quo Policy")

    # Add coastlines and save figure
    output_file = "Stroke-Triage-Policy-Project Updated/PlottingFiles/grid_plot_$(option).pdf"
    coast!(region=my_region, savefig=output_file, show=true, proj="merc")
    println("Saved plot to: $output_file")
end

# Compute probabilities from per-cell CSV files
println("Computing probabilities from per-cell data...")
probabilities_best, probabilities_ca = compute_probabilities(GRID_SIZE, INPUT_DIR)

# Generate plots for "CA" and "smarter" policies
println("Generating 'CA' plot...")
make_plot_with_grid(GRID_SIZE, MY_REGION, probabilities_ca, "CA")

println("Generating 'smarter' plot...")
make_plot_with_grid(GRID_SIZE, MY_REGION, probabilities_best, "smarter")

println("Plots generated successfully!")
