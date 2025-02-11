

using CSV
using Glob

# Function to check if the first line of a CSV file matches a condition
function delete_csv_files_by_first_line(directory::String, condition::String)
    # Get a list of all CSV files in the directory
    csv_files = glob("*.csv", directory)

    if !isdir(directory)
        println("Error: The directory '$directory' does not exist.")
        return
    end

    for file in csv_files
        try
            # Read the first row of the CSV file
            println("hello")
            first_row = CSV.File(file, header=false) |> first
            first_line = string(first_row...)

            # Check if the first line matches the condition
            if startswith(first_line, condition)
                println("Deleting file: $file")
                rm(file)  # Delete the file
            end
        catch e
            println("Error reading file $file: $e")
        end
    end
end

directory = "Stroke-Triage-Policy-Project Updated/PlottingFiles/grid_plot_csvs_CA"  # Replace with the path to your directory
condition = "error"  # Replace with the string you're looking for in the first line
delete_csv_files_by_first_line(directory, condition)