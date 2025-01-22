using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, 
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params, conc
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise
using ATNModelling.InferenceModels: fit_model, ensemble_atn_harmonised, serial_atn, fit_serial_atn

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
n_samples = parse(Int, ARGS[2])
n_chains = parse(Int, ARGS[3])

# --------------------------------------------------------------------------------
# Tracer independent data
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c) 
dktnames = get_parcellation() |> get_cortex |> get_dkt_names

# Amyloid data 
_ab_data_df =  CSV.read(datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 &&  x.NEO_Status == 1, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=3)
# --------------------------------------------------------------------------------
# Load fbb data
# --------------------------------------------------------------------------------
tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

fbb, fbb_tau = align_data(fbb_data, tau_data; min_tau_scans=3)

fbb_times = get_times.(fbb)
fbb_tau_times = get_times.(fbb_tau)
fbb_ts = [sort(unique([a; t])) for (a, t) in zip(fbb_times, fbb_tau_times)]

fbb_ab_tidx = get_time_idx(fbb_times, fbb_ts)
fbb_tau_tidx = get_time_idx(fbb_tau_times, fbb_ts)

@assert get_id.(fbb) == get_id.(fbb_tau)
@assert allequal([allequal(fbb_times[i] .== fbb_ts[i][fbb_ab_tidx[i]]) for i in 1:length(fbb)])
@assert allequal([allequal(fbb_tau_times[i] .== fbb_ts[i][fbb_tau_tidx[i]]) for i in 1:length(fbb_tau)])

fbb_suvr = calc_suvr.(fbb)
normalise!(fbb_suvr, fbb_u0, fbb_ui)
fbb_conc = map(x -> conc.(x, fbb_u0, fbb_ui), fbb_suvr)
fbb_inits = [d[:,1] for d in fbb_conc]

fbb_tau_suvr = calc_suvr.(fbb_tau)
normalise!(fbb_tau_suvr, v0, vi)
fbb_tau_conc = map(x -> conc.(x, v0, vi), fbb_tau_suvr)
fbb_tau_inits = [d[:,1] for d in fbb_tau_conc]

fbb_tau_pos_vol = get_vol.(fbb_tau)
fbb_total_vol_norm = [tp ./ sum(tp, dims=1) for tp in fbb_tau_pos_vol]
fbb_vols = [clamp.(1 .- (vol ./ vol[:,1]), 0, 1) for vol in fbb_total_vol_norm]
fbb_vol_inits = [vol[:,1] for vol in fbb_vols]

fbb_inits = [[ab; tau; vol] for (ab, tau, vol) in zip(fbb_inits, fbb_tau_inits, fbb_vol_inits)]
fbb_n = length(fbb)

fbb_atn_model = make_scaled_atn_model(fbb_ui .- fbb_u0, part .- v0, L)
fbb_prob = make_prob(fbb_atn_model, fbb_inits[1], (0.0,7.5), [1.0,0.1,1.0,3.5,1.0])
# --------------------------------------------------------------------------------
# Load fbp data
# --------------------------------------------------------------------------------
tracer="FBP"
fbp_u0, fbp_ui = load_ab_params(tracer=tracer)
fbp_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
fbp_data = ADNIDataset(fbp_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

fbp, fbp_tau = align_data(fbp_data, tau_data; min_tau_scans=3)

fbp_times = get_times.(fbp)
fbp_tau_times = get_times.(fbp_tau)
fbp_ts = [sort(unique([a; t])) for (a, t) in zip(fbp_times, fbp_tau_times)]

fbp_ab_tidx = get_time_idx(fbp_times, fbp_ts)
fbp_tau_tidx = get_time_idx(fbp_tau_times, fbp_ts)

@assert get_id.(fbp) == get_id.(fbp_tau)
@assert allequal([allequal(fbp_times[i] .== fbp_ts[i][fbp_ab_tidx[i]]) for i in 1:length(fbp)])
@assert allequal([allequal(fbp_tau_times[i] .== fbp_ts[i][fbp_tau_tidx[i]]) for i in 1:length(fbp_tau)])

fbp_suvr = calc_suvr.(fbp)
normalise!(fbp_suvr, fbp_u0, fbp_ui)
fbp_conc = map(x -> conc.(x, fbp_u0, fbp_ui), fbp_suvr)
fbp_inits = [d[:,1] for d in fbp_conc]

fbp_tau_suvr = calc_suvr.(fbp_tau)
normalise!(fbp_tau_suvr, v0, vi)
fbp_tau_conc = map(x -> conc.(x, v0, vi), fbp_tau_suvr)
fbp_tau_inits = [d[:,1] for d in fbp_tau_conc]

fbp_tau_pos_vol = get_vol.(fbp_tau)
fbp_total_vol_norm = [tp ./ sum(tp, dims=1) for tp in fbp_tau_pos_vol]
fbp_vols = [clamp.(1 .- (vol ./ vol[:,1]), 0, 1) for vol in fbp_total_vol_norm]
fbp_vol_inits = [vol[:,1] for vol in fbp_vols]

fbp_inits = [[ab; tau; vol] for (ab, tau, vol) in zip(fbp_inits, fbp_tau_inits, fbp_vol_inits)]
fbp_n = length(fbp)

fbp_atn_model = make_scaled_atn_model(fbp_ui .- fbp_u0, part .- v0, L)
fbp_prob = make_prob(fbp_atn_model, fbp_inits[1], (0.0,7.5), [1.0,0.1,1.0,3.5,1.0])

# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
fbb_vec_data = vectorise(fbb_conc)
fbb_tau_vec_data = vectorise(fbb_tau_conc)
fbb_vol_vec_data = vectorise(fbb_vols)

fbp_vec_data = vectorise(fbp_conc)
fbp_tau_vec_data = vectorise(fbp_tau_conc)
fbp_vol_vec_data = vectorise(fbp_vols)


@assert allequal(0 .<= fbb_vec_data .<= 1)
@assert allequal(0 .<= fbb_tau_vec_data .<= 1)
@assert allequal(0 .<= fbb_vol_vec_data .<= 1)

@assert allequal(0 .<= fbp_vec_data .<= 1)
@assert allequal(0 .<= fbp_tau_vec_data .<= 1)
@assert allequal(0 .<= fbp_vol_vec_data .<= 1)

fbb_idx = 1:16
fbp_idx = 17:24
n = 24

Random.seed!(1234)

m = ensemble_atn_harmonised(fbb_prob, fbb_inits, fbb_ts, fbb_ab_tidx, fbb_tau_tidx, fbb_idx, fbb_n,
                        fbp_prob, fbp_inits, fbp_ts, fbp_ab_tidx, fbp_tau_tidx, fbp_idx, fbp_n, n)

pst = m | (fbb_data = fbb_vec_data, fbb_tau_data = fbb_tau_vec_data, fbb_vol_data = fbb_tau_vec_data,
          fbp_data = fbp_vec_data, fbp_tau_data = fbp_tau_vec_data, fbp_vol_data = fbp_tau_vec_data,);
pst()

println("Starting Inference")
samples = sample(pst, NUTS(0.8), MCMCSerial(), n_samples, n_chains)
println("Number of Divergences: $(sum(samples[:numerical_error]))")
display(summarize(samples))

serialize(projectdir("output/chains/population-scaled-atn/pst-samples-harmonised-$(tracer)-$(n_chains)x$(n_samples).jls"), pst)