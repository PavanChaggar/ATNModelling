using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, 
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params, concentration
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise
using ATNModelling.InferenceModels: fit_model, ensemble_atn_truncated, serial_atn, fit_serial_atn

using Connectomes: laplacian_matrix, get_label
using ADNIDatasets: ADNIDataset, get_id, get_dates, get_initial_conditions, calc_suvr, get_vol, get_times
using DrWatson: projectdir, datadir
using CSV, DataFrames
using SciMLBase: successful_retcode
using DifferentialEquations, Turing, LinearAlgebra
using ADTypes: AutoZygote
using Random
using SciMLSensitivity
using Serialization
# --------------------------------------------------------------------------------
# Script params 
# --------------------------------------------------------------------------------
tracer = ARGS[1]
n_samples = parse(Int, ARGS[2])
n_chains = parse(Int, ARGS[3])
# --------------------------------------------------------------------------------
# Load parameters
# --------------------------------------------------------------------------------
u0, ui = load_ab_params(tracer=tracer)
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

ab_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
# ab_data_df = filter(x -> x.qc_flag==2, _ab_data_df);
ab_data = ADNIDataset(ab_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

# Tau data 
tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=3)

ab, tau = align_data(ab_data, tau_data; min_tau_scans=3)

ab_times = get_times.(ab)
tau_times = get_times.(tau)
ts = [sort(unique([a; t])) for (a, t) in zip(ab_times, tau_times)]

ab_tidx = get_time_idx(ab_times, ts)
tau_tidx = get_time_idx(tau_times, ts)

@assert get_id.(ab) == get_id.(tau)
@assert allequal([allequal(ab_times[i] .== ts[i][ab_tidx[i]]) for i in 1:length(ab)])
@assert allequal([allequal(tau_times[i] .== ts[i][tau_tidx[i]]) for i in 1:length(tau)])

ab_suvr = calc_suvr.(ab)
normalise!(ab_suvr, u0, ui)
ab_conc = map(x -> conc.(x, u0, ui), ab_suvr)
ab_inits = [d[:,1] for d in ab_conc]

tau_suvr = calc_suvr.(tau)
normalise!(tau_suvr, v0, vi)
tau_conc = map(x -> conc.(x, v0, vi), tau_suvr)
tau_inits = [d[:,1] for d in tau_conc]

tau_pos_vol = get_vol.(tau)
total_vol_norm = [tp ./ sum(tp, dims=1) for tp in tau_pos_vol]
vols = [clamp.(1 .- (vol ./ vol[:,1]), 0, 1) for vol in total_vol_norm]
vol_inits = [vol[:,1] for vol in vols]

atn_model = make_scaled_atn_model(ui .- u0, part .- v0, L)
prob = make_prob(atn_model, 
          [ab_inits[1]; tau_inits[1]; vol_inits[1]], 
          (0.0,7.5), [1.0,0.1,1.0,3.5,1.0])
sol = solve(prob, Tsit5())

inits = [[ab; tau; vol] for (ab, tau, vol) in zip(ab_inits, tau_inits, vol_inits)]
n_subjects = length(ab)

# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
ab_vec_data = vectorise(ab_conc)
tau_vec_data = vectorise(tau_conc)
vol_vec_data = vectorise(vols)

@assert allequal(0 .<= ab_vec_data .<= 1)
@assert allequal(0 .<= tau_vec_data .<= 1)
@assert allequal(0 .<= vol_vec_data .<= 1)

Random.seed!(1234)
pst = fit_model(ensemble_atn_truncated, ab_vec_data, tau_vec_data, vol_vec_data, 
                     prob, inits, ts, ab_tidx, tau_tidx, n_subjects;
                     n_samples=n_samples, n_chains=n_chains)

serialize(projectdir("output/chains/population-scaled-atn/pst-samples-$(tracer)-$(n_chains)x$(n_samples).jls"), pst)