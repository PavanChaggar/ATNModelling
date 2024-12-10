using ATNModelling.SimulationUtils: load_ab_params, simulate_amyloid
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: normalise!
using ATNModelling.InferenceModels: ab_inference, population_ab_inference, fit_ab_model

using ADNIDatasets: ADNIDataset, get_id, get_times, get_initial_conditions, calc_suvr
using DrWatson: projectdir, datadir
using CSV, DataFrames
using Serialization, Dates
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
tau_pos = ADNIDataset(tau_pos_df, dktnames; min_scans=3)

pos_ids = get_id.(tau_pos)
tau_pos_idx = reduce(vcat, [findall(x -> get_id(x) ∈ id, ab_data) for id in pos_ids])

ab_tau_pos = ab_data[tau_pos_idx]
ab_suvr = calc_suvr.(ab_tau_pos)
ab_times = get_times.(ab_tau_pos)

# --------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------
u0, ui = load_ab_params()
ui_diff = ui .- u0

n_subjects = length(ab_tau_pos)

n_samples = 1000
n_chains  = 4
pst_samples = fit_ab_model(population_ab_inference, 
                          ab_suvr, 
                          ab_inits, u0, ui_diff, ab_times, n_subjects; 
                          n_samples = n_samples, n_chains = n_chains)


serialize(
    projectdir("output/chains/population-ab/pst-population-ab-$(n_chains)x$(n_samples).jls"), 
    pst_samples)