using ATNModelling.SimulationUtils: make_prob, make_atn_feedback_model, 
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise
using ATNModelling.InferenceModels: fit_model, ensemble_atn, serial_atn, fit_serial_atn

using Connectomes: laplacian_matrix, get_label
using ADNIDatasets: ADNIDataset, get_id, get_dates, get_initial_conditions, calc_suvr, get_vol, get_times
using DrWatson: projectdir, datadir
using CSV, DataFrames
using SciMLBase: successful_retcode
using DifferentialEquations, Turing, LinearAlgebra
using Random
# --------------------------------------------------------------------------------
# Load parameters
# --------------------------------------------------------------------------------
u0, ui = load_ab_params()
ui_diff = ui .- u0
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c)

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

ab_times = get_times.(ab)
tau_times = get_times.(tau)
ts = [sort(unique([a; t])) for (a, t) in zip(ab_times, tau_times)]

ab_tidx = get_time_idx(ab_times, ts)
tau_tidx = get_time_idx(tau_times, ts)

@assert allequal([allequal(ab_times[i] .== ts[i][ab_tidx[i]]) for i in 1:22])
@assert allequal([allequal(tau_times[i] .== ts[i][tau_tidx[i]]) for i in 1:22])

ab_suvr = calc_suvr.(ab)
normalise!(ab_suvr, u0, ui)
ab_inits = [d[:,1] for d in ab_suvr]

tau_suvr = calc_suvr.(tau)
normalise!(tau_suvr, v0)
tau_inits = [d[:,1] for d in tau_suvr]

tau_pos_vol = get_vol.(tau)
total_vol_norm = [tp ./ sum(tp, dims=1) for tp in tau_pos_vol]
vols = [clamp.(1 .- (vol ./ vol[:,1]), 0, 1) for vol in total_vol_norm]
vol_inits = [vol[:,1] for vol in vols]

atn_model = make_atn_feedback_model(u0, ui, v0, part, L)
prob = make_prob(atn_model, 
          [ab_inits[1]; tau_inits[1]; vol_inits[1]], 
          (0.0,7.5), [1.0,1.0,1.0,3.5,0.1])

inits = [[ab; tau; vol] for (ab, tau, vol) in zip(ab_inits, tau_inits, vol_inits)]
n_subjects = length(ab)

# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
ab_vec_data = vectorise(ab_suvr)
tau_vec_data = vectorise(tau_suvr)
vol_vec_data = vectorise(vols)

Random.seed!(1234)
n_samples = 1000
n_chains = 4
pst = fit_model(ensemble_atn_truncated, ab_vec_data, tau_vec_data, vol_vec_data, prob, inits, ts, ab_tidx, tau_tidx, n_subjects; 
                n_samples=1000, n_chains=4)

using Serialization
serialize(projectdir("output/chains/population-atn/pst-samples-truncated-normal-feedback-$(n_chains)x$(n_samples).jls"), pst)
