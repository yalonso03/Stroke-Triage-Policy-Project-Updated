#=
Modeling Stroke Patient Triage and Transport in CA with MDPs
Autumn 2023 CS199/CS195 Project
Emily Molins and Yasmine Alonso

File: action_enum.jl
--------------------
Hacky way of doing things, but since you can't define an enum at runtime this was the most effective way to do it... 
Just feed in the filename of the CSV to hospital_file. It'll print to the console, line by line, what all the actions should be

i.e. ROUTE_[Hospital Name]

Simply paste in the output into the definition of the Action enum in STPMDP.jl
=#

include("STPMDP.jl")


hospital_file = "HospitalInfo/hospital_list_RI.csv"
locs = csv_to_locations(hospital_file)  # defined in STPMDP.jl
for loc in locs
    println("ROUTE_" * loc.name)
end
