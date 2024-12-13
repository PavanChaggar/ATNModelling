using ATNModelling.SimulationUtils: load_ab_params, load_tau_params
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx
using Connectomes
using GLMakie, Colors, ColorSchemes
using ADNIDatasets: ADNIDataset, get_id, get_dates, 
                    get_initial_conditions, calc_suvr, get_vol, get_times,
                    data_dashboard
using DrWatson: projectdir, datadir
using CSV, DataFrames

# --------------------------------------------------------------------------------
# Load parameters
# --------------------------------------------------------------------------------
u0, ui = load_ab_params()
ui_diff = ui .- u0
v0, vi, part = load_tau_params()

# --------------------------------------------------------------------------------
# Loading data and aligning
# --------------------------------------------------------------------------------
dktnames = get_parcellation() |> get_cortex |> get_dkt_names

# Amyloid data 
_ab_data_df =  CSV.read(datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame) 

ab_data_df = filter(x -> x.qc_flag==2 && x.TRACER == "FBP", _ab_data_df);
ab_data = ADNIDataset(ab_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

# Tau data 
tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=3)

ab, tau = align_data(ab_data, tau_data)

data_dashboard(ab, fill(minimum(u0), 72), fill(maximum(ui), 72); cmap=ColorSchemes.viridis, show_mtl_threshold=false)
data_dashboard(tau, fill(minimum(v0), 72), fill(maximum(vi), 72); cmap=ColorSchemes.viridis, show_mtl_threshold=true)
