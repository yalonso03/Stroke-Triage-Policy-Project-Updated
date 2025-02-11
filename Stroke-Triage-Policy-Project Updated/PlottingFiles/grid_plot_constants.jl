#=
Modeling Stroke Patient Triage and Transport in CA with MDPs
Jan 2025
Emily Molins and Yasmine Alonso

File: grid_plot_constants.jl
----------------------------
This file contains various constants needed by both grid_plot_data_CA.jl and grid_plot_graph_CA.jl which generate sample patient
datapoints to produce a hotspot plot over the bay area.
=#

GRID_SIZE = 200
HAYWARD_LAT = 37.6688
HAYWARD_LON = -122.0808
ZOOM_LEVEL = 0.5
MY_REGION = [HAYWARD_LON - ZOOM_LEVEL, HAYWARD_LON + ZOOM_LEVEL, HAYWARD_LAT - ZOOM_LEVEL, HAYWARD_LAT + ZOOM_LEVEL]
OUTPUT_DIR = "Stroke-Triage-Policy-Project Updated/PlottingFiles/grid_plot_csvs_CA"