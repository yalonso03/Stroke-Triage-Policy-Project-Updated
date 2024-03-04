include("STPMDP.jl")

using GMT
using CSV

function get_hospital_locations(infile)
    CSCs = []
    PSCs = []
    clinics = []
    df = CSV.read(infile, DataFrame, delim=',')
    for row in eachrow(df)
        lat = row["Lat"]
        lon = row["Lon"]
        tup = (lat, lon)
        if row["Type"] == "CLINIC"
            push!(clinics, tup)
        elseif row["Type"] == "CSC"
            push!(CSCs, tup)
        else
            push!(PSCs, tup)
        end
    end
    return CSCs, PSCs, clinics
end

function parse_coordinates(coord_string)
    # Remove parentheses and split the string into latitude and longitude parts
    parts = split(strip(coord_string, ['(', ')']), ", ")
    
    # Parse latitude and longitude as floats
    latitude = parse(Float64, parts[1])
    longitude = parse(Float64, parts[2])
    
    # Return a tuple of floats
    return (latitude, longitude)
end


struct LatlonAndProb
    latlon::Tuple{Float64, Float64}
    prob::Float64
end

#best_action_reward%nearest_hospital_reward%start_state_latlon
# Takes in a simulation result file, and creates two vectors (one for california's policy, one for ours), which are vectors of LatlonAndProb
# Structs, which simply store a latlon tuple (representing the starting location of the patient in the field) and the probability of a good Outcome
# that was determined. hopefully the probability of a good outcome with our policy is greater than the default 
function get_patient_locations(infile)
    df = CSV.read(infile, DataFrame, delim='%')
    our_policy = []
    CA_policy = []
    for row in eachrow(df)
        best_a_reward = Float64(row["best_action_reward"])
        best_a_latlon = row["start_state_latlon"]
        latlon_tuple_best = parse_coordinates(best_a_latlon)
        push!(our_policy, LatlonAndProb(latlon_tuple_best, best_a_reward))

        nearest_hosp_reward = Float64(row["nearest_hospital_reward"])
        nearest_hosp_latlon = row["start_state_latlon"]
        latlon_tuple_nearest = parse_coordinates(nearest_hosp_latlon)
        push!(CA_policy, LatlonAndProb(latlon_tuple_nearest, nearest_hosp_reward))
    end
    return our_policy, CA_policy
end

function make_plot_our_policy(hospital_csv, simulations_csv)
    # Define the latitude and longitude of Hayward, CA
    hayward_lat = 37.6688
    hayward_lon = -122.0808
    
    # Define the zoom level (adjust this as needed)
    zoom_level = 0.5  # You can adjust this value to zoom in or out

    # Calculate the region coordinates based on Hayward's coordinates and zoom level
    region = [hayward_lon - zoom_level, hayward_lon + zoom_level, hayward_lat - zoom_level, hayward_lat + zoom_level]

    coast(region=region, proj=:Mercator, frame=:n, area=1000, land=:gray, water=:white, grid=false)

    CSCs, PSCs, clinics = get_hospital_locations(hospital_csv)
    our_policy_pts, CA_policy_pts = get_patient_locations(simulations_csv)

    CSC_lons = [tup[1] for tup in CSCs]
    CSC_lats = [tup[2] for tup in CSCs]

    PSC_lons = [tup[1] for tup in PSCs]
    PSC_lats = [tup[2] for tup in PSCs]

    clinics_lons = [tup[1] for tup in clinics]
    clinics_lats = [tup[2] for tup in clinics]

    
    our_policy_pts_lats = [pt.latlon[2] for pt in our_policy_pts]
    our_policy_pts_lons = [pt.latlon[1] for pt in our_policy_pts]
    our_policy_rewards = [pt.prob for pt in our_policy_pts]
    
    # Plot our routing policy as orange cirlces (for now, color will change later)
    GMT.scatter!(
        colorbar=(name="Colorbar"),
        our_policy_pts_lats, 
        our_policy_pts_lons, 
        fmt=:png, 
        marker=:circle,
        color=:red2green,
        alpha=50,
        zcolor=our_policy_rewards,
        legend="Patient Outcomes",
        markeredgecolor=:black,
        size=0.15 
        #markerfacecolor=:orange
    )

    
    # Plot the CSCs as green squares
    GMT.scatter!(
        CSC_lats, 
        CSC_lons, 
        fmt=:png, 
        marker=:square,
        markeredgecolor=:black,
        legend="CSCs",
        size=0.1,  
        markerfacecolor=:green
    )

    # Plot the PSCs as magenta squares
    GMT.scatter!(
        PSC_lats, 
        PSC_lons, 
        fmt=:png, 
        marker=:square,
        legend="PSCs",
        markeredgecolor=:black,
        size=0.1,  
        markerfacecolor=:blue
        
    )

    # Plot the clinics as purple squares
    GMT.scatter!(
        clinics_lats, 
        clinics_lons, 
        fmt=:png, 
        marker=:square,
        markeredgecolor=:black,
        legend="Clinics",
        size=0.1,  
        markerfacecolor=:purple,
        title="Optimal Policy Results",
        show=true,
        savefig="Figures/Optimal_Final.pdf"
    )
    
end


function make_plot_CA_policy(hospital_csv, simulations_csv)
    # Define the latitude and longitude of Hayward, CA
    hayward_lat = 37.6688
    hayward_lon = -122.0808
    
    # Define the zoom level (adjust this as needed)
    zoom_level = 0.5  # You can adjust this value to zoom in or out

    # Calculate the region coordinates based on Hayward's coordinates and zoom level
    region = [hayward_lon - zoom_level, hayward_lon + zoom_level, hayward_lat - zoom_level, hayward_lat + zoom_level]

    coast(region=region, proj=:Mercator, frame=:n, area=1000, land=:gray, water=:white, grid=false)
    CSCs, PSCs, clinics = get_hospital_locations(hospital_csv)
    our_policy_pts, CA_policy_pts = get_patient_locations(simulations_csv)

    CSC_lons = [tup[1] for tup in CSCs]
    CSC_lats = [tup[2] for tup in CSCs]

    PSC_lons = [tup[1] for tup in PSCs]
    PSC_lats = [tup[2] for tup in PSCs]

    clinics_lons = [tup[1] for tup in clinics]
    clinics_lats = [tup[2] for tup in clinics]

    CA_policy_pts_lats = [(pt.latlon)[2] for pt in CA_policy_pts]
    CA_policy_pts_lons = [(pt.latlon)[1] for pt in CA_policy_pts]
    CA_policy_rewards = [pt.prob for pt in CA_policy_pts]

    # Plot california's routing policy as black circles (for now, color will change later)
    GMT.scatter!(
        # colorbar=(name="Colorbar"),
        CA_policy_pts_lats, 
        CA_policy_pts_lons, 
        fmt=:png, 
        marker=:circle,
        markeredgecolor=:black,
        color=:red2green,
        legend="Patient Outcomes",
        alpha=50,
        zcolor=CA_policy_rewards,
        size=0.15
    )

    # Plot the CSCs as green squares
    GMT.scatter!(
        CSC_lats, 
        CSC_lons, 
        fmt=:png, 
        marker=:square,
        markeredgecolor=:black,
        legend="CSCs",
        size=0.1,  
        markerfacecolor=:green
    )

    # Plot the PSCs as magenta squares
    GMT.scatter!(
        PSC_lats, 
        PSC_lons, 
        fmt=:png, 
        marker=:square,
        legend="PSCs",
        markeredgecolor=:black,
        size=0.1,  
        markerfacecolor=:blue
    )

    # Plot the clinics as purple squares
    GMT.scatter!(
        clinics_lats, 
        clinics_lons, 
        fmt=:png, 
        marker=:square,
        markeredgecolor=:black,
        legend="Clinics",
        size=0.1,  
        markerfacecolor=:purple,
        title="Status Quo Policy Results",
        show=true,
        savefig="Figures/Status_Quo_Final.pdf"
    ) 
end


hospital_csv = "HospitalInfo/hospital_list_CA.csv"
sims_csv = "SimResults/1000points.csv"
make_plot_our_policy(hospital_csv, sims_csv)
make_plot_CA_policy(hospital_csv, sims_csv)


