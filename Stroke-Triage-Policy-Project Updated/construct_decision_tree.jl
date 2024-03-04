using CSV
using DataFrames
using DecisionTree
using AbstractTrees
using TikzGraphs
using Graphs
using TikzPictures


include("STPMDP.jl")


N_SAMPLES = 1000

# ROUTE_KaiserPermanenteRedwoodCity

function rand_location()
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

function hospital_type(a::Action)
    a_str = enum_to_string(a)
    
    # Now we have "ROUTE_KaiserPermanenteRedwoodCity" as a string, extract "KaiserPermanenteRedwoodCity"replace(original_string, "ROUTE_" => "")
    hospital_name = replace(a_str, "ROUTE_" => "")

    # Search for the hospital name in the CSV to determine LocType CSC, PSC, or Clinic
    csv_file_path = "HospitalInfo/hospital_list_CA.csv"
    df = CSV.File(csv_file_path) |> DataFrame  # make a dataframe of the CSV
    row_index = findfirst(df[:, :Hospital] .== hospital_name)

    if row_index !== nothing
        # get CSC, PSC, or Clinic
        hospital_type = df[row_index, :Type]
        
    else
        error("Tried to find a hospital that did not exist!")
    end
    return eval(Meta.parse(hospital_type))  # turns "CSC" into LocType CSC
end

# For our features
time_to_CSCs::Vector{Float64} = []
time_to_PSCs::Vector{Float64}  = []
time_to_clinics::Vector{Float64}  = []
t_onsets::Vector{Float64}  = []

labels = []

mdp = StrokeMDP()


for i in 1:N_SAMPLES
    # Sample a random PatientState
    sampled_s = PatientState(Location("FIELD1", rand_location(), -1, FIELD), rand() * 270, UNKNOWN, sample_stroke_type(mdp))

    # Gather desired information about the features we will use to fit our tree
    nearest_CSC = find_nearest_CSC(mdp, sampled_s.loc)
    t_nearest_CSC = calculate_travel_time(sampled_s.loc, nearest_CSC)
    
    nearest_PSC = find_nearest_PSC(mdp, sampled_s.loc)
    t_nearest_PSC = calculate_travel_time(sampled_s.loc, nearest_PSC)

    nearest_clinic = find_nearest_clinic(mdp, sampled_s.loc)
    t_nearest_clinic = calculate_travel_time(sampled_s.loc, nearest_clinic)

    push!(time_to_CSCs, t_nearest_CSC)
    push!(time_to_PSCs, t_nearest_PSC)
    push!(time_to_clinics, t_nearest_clinic)
    push!(t_onsets, sampled_s.t_onset)

    # Now, get our labels by using our smarter policy (forward search) to a depth of 2
    a = best_action(mdp, sampled_s, 2)

    # But, we want our labels to be more general (route to nearest CSC, route to nearest PSC, route to nearest Clinic)
    # 1 means route to CSC nearest
    # 2 means route to PSC nearest
    # 3 means route to Clinic nearest
    type = hospital_type(a)
    if type == CSC
        label = 1
    elseif type == PSC
        label = 2
    else
        label = 3
    end

    push!(labels, label)
end

# new features -- comment out if going with option 2
diffs_CSC_PSC::Vector{Float64} = [abs(time_to_CSCs[i] - time_to_PSCs[i]) for i=1:N_SAMPLES]
diffs_CSC_Clinic::Vector{Float64} = [abs(time_to_CSCs[i] - time_to_clinics[i]) for i=1:N_SAMPLES]
diffs_PSC_Clinic::Vector{Float64} = [abs(time_to_PSCs[i] - time_to_clinics[i]) for i=1:N_SAMPLES]

ratios_CSC_PSC::Vector{Float64} = [time_to_CSCs[i] / time_to_PSCs[i] for i=1:N_SAMPLES]
ratios_CSC_Clinic::Vector{Float64} = [time_to_CSCs[i] / time_to_clinics[i] for i=1:N_SAMPLES]
ratios_PSC_Clinic::Vector{Float64} = [time_to_PSCs[i] / time_to_clinics[i] for i=1:N_SAMPLES]

# Option 1: more features
features = hcat(time_to_CSCs, time_to_PSCs, time_to_clinics, t_onsets, diffs_CSC_PSC, diffs_CSC_Clinic, diffs_PSC_Clinic, ratios_CSC_PSC, ratios_CSC_Clinic, ratios_PSC_Clinic)

# Option 2: just og features
features = hcat(time_to_CSCs, time_to_PSCs, time_to_clinics, t_onsets)


model = DecisionTreeClassifier(max_depth=4)
DecisionTree.fit!(model, features, labels)


DecisionTree.print_tree(model)

#=
Notes from mykel

- Save all of the training data into a CSV, open in excel and sanity Check
- Check what's yes and what's no again to be safe for the branching
- Run simulations where the policy is making the decisions -- make a fn representing the tree 
=#



#=
# Citation: Robert Moss!
# All of the below functions are taken from his pluto notebook. 
typeof(DecisionTree.wrap(model.root, (featurenames=state_labels, classlabels=action_labels)))

TikzGraphs.edge_str(g) = "--"

function TikzGraphs.plot(g; layout::Layouts.Layout=Layouts.Layered(), labels::Vector{T}=map(string, vertices(g)), edge_labels::Dict = Dict(), node_styles::Dict = Dict(), node_style="", edge_styles::Dict = Dict(), edge_style="", options="", graph_options="", prepend_preamble::String="") where T<:AbstractString
    o = IOBuffer()
    println(o, "\\graph [$(TikzGraphs.layoutname(layout)), $(TikzGraphs.options_str(layout)), $graph_options] {")
    for v in vertices(g)
        TikzGraphs.nodeHelper(o, v, labels, node_styles, node_style)
    end
    println(o, ";")
    for e in edges(g)
        a = src(e)
        b = dst(e)
        print(o, "$a $(TikzGraphs.edge_str(g))")
        TikzGraphs.edgeHelper(o, a, b, edge_labels, edge_styles, edge_style)
        println(o, "$b;")
    end
    println(o, "};")
    mypreamble = prepend_preamble * TikzGraphs.preamble * "\n\\usegdlibrary{$(TikzGraphs.libraryname(layout))}"
    TikzGraphs.TikzPicture(String(take!(o)), preamble=mypreamble, options=options)
end

Base.length(tree::InfoNode) = 1 + sum(length(child) for child in children(tree))

Base.length(leaf::InfoLeaf) = 1

function node2str(leaf::InfoLeaf; rounding=false, sigdigits=3)
	if hasproperty(leaf, :info) && hasproperty(leaf.info, :classlabels)
		return string(leaf.info.classlabels[leaf.leaf.majority])
	else
		majority = leaf.leaf.majority
		if rounding
			return string(round(majority; sigdigits))
		else
			return string(majority)
		end
	end
end


function tree2graph(model::Union{DecisionTreeClassifier,DecisionTreeRegressor}, features, classes; rounding=false, sigdigits=3)
    tree = DecisionTree.wrap(model.root, (featurenames=features, classlabels=classes))
    g, tree_labels = tree2graph(tree; rounding, sigdigits)
    return TikzGraphs.plot(g; labels=tree_labels)
end

function tree2graph(model::Union{DecisionTreeClassifier,DecisionTreeRegressor}, features; rounding=false, sigdigits=3)
    tree = DecisionTree.wrap(model.root, (featurenames=features,))
    g, tree_labels = tree2graph(tree; rounding, sigdigits)
    return TikzGraphs.plot(g; labels=tree_labels)
end

function tree2graph(tree::InfoNode, g=SimpleGraph(length(tree)), ids=[1], labels=[]; rounding=false, sigdigits=3)
	if isempty(labels)
		labels = [node2str(tree; rounding, sigdigits)]
	end
	i_root = ids[end]
	for child in children(tree)
		push!(ids, length(ids)+1)
		push!(labels, node2str(child; rounding, sigdigits))
		add_edge!(g, i_root, ids[end])
		tree2graph(child, g, ids, labels; rounding, sigdigits)
	end
	return g, labels
end

function node2str(tree::InfoNode; rounding=false, sigdigits=3)
	val = tree.node.featval
	if rounding
		val = round(val; sigdigits)
	end
	return string(tree.info.featurenames[tree.node.featid], " < ", val)
end

tree2graph(tree::InfoLeaf, g=SimpleGraph(length(tree)), ids=[], labels=[]; kwargs...) = nothing


#! below code would print out the tree more nicely. Still trying to get it to work. 
state_labels = ["Time to CSC", "Time to PSC", "Time to Clinic", "Time since symptom onset"]
action_labels = ["Route_CSC", "Route_PSC", "Route_Clinic"]
# tikz = tree2graph(model, state_labels, action_labels)
# TikzGraphs.TikzPictures.save(TikzGraphs.TEX("ld_dtree"), tikz)
=#