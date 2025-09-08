using ATNModelling.SimulationUtils: make_prob, make_atn_model, 
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex_parc, get_dkt_names
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise
using ATNModelling.InferenceModels: fit_model, ensemble_atn_truncated, serial_atn, fit_serial_atn

using Connectomes: laplacian_matrix, get_label
using ADNIDatasets: ADNIDataset, get_id, get_dates, get_initial_conditions, calc_suvr, get_vol, get_times
using DrWatson: projectdir, datadir
using CSV, DataFrames
using SciMLBase: successful_retcode
using DifferentialEquations, Turing, LinearAlgebra
using Random
include(projectdir("bf-data.jl"))
# --------------------------------------------------------------------------------
# Load parameters
# --------------------------------------------------------------------------------
u0, ui = load_ab_params(tracer="FMM")
ui_diff = ui .- u0
v0, vi, part = load_tau_params(tracer="RO")
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c)

# --------------------------------------------------------------------------------
# Loading data and aligning
# --------------------------------------------------------------------------------
dktnames = get_parcellation() |> get_cortex_parc |> get_dkt_names

data_path = datadir("bf-data/bf-data-ab-tau-summary.csv");

data_df = CSV.read(data_path, DataFrame)

dktnames = get_parcellation() |> get_cortex_parc |> get_dkt_names

ab_data_df = filter(x -> x.ab_status == 1, data_df)
ab_data = BFDataset(ab_data_df, dktnames; min_scans=3, tracer=:ab)

# Tau data 
tau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, ab_data_df);
tau_data = BFDataset(tau_pos_df, dktnames; min_scans=3, tracer=:tau)

ab, tau = align_data(ab_data, tau_data)

ab_times = get_times.(ab)
tau_times = get_times.(tau)
ts = [sort(unique([a; t])) for (a, t) in zip(ab_times, tau_times)]

ab_tidx = get_time_idx(ab_times, ts)
tau_tidx = get_time_idx(tau_times, ts)

@assert allequal([allequal(ab_times[i] .== ts[i][ab_tidx[i]]) for i in 1:38])
@assert allequal([allequal(tau_times[i] .== ts[i][tau_tidx[i]]) for i in 1:38])

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

atn_model = make_atn_model(u0, ui, v0, part, L)
prob = make_prob(atn_model, 
          [ab_inits[1]; tau_inits[1]; vol_inits[1]], 
          (0.0,7.5), [1.0,1.0,1.0,3.5,1.0])

inits = [[ab; tau; vol] for (ab, tau, vol) in zip(ab_inits, tau_inits, vol_inits)]
n_subjects = length(ab)

# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
using LsqFit

linearmodel(x, p) =  part .+ p[1] * x
fitted_model = curve_fit(linearmodel, ui .- u0, vi, [1.0])
fitted_model.param

# linearmodel(x, p) = part .+ p[1] .* x
# fbb_fitted_model = curve_fit(linearmodel, fbb_ui .- fbb_u0, vi, [1.0])
# using CairoMakie
# begin
#     f = Figure(size=(600, 500))
#     ax = Axis(f[1,1])
#     ylims!(ax, 0.0, 5.0)
#     xlims!(ax, 0.0, 5.0)
#     scatter!(ui .- u0, vi .- part)
#     lines!(0:0.1:1.5, linearmodel(0:0.1:1.5, fitted_model.param[1]))
#     # scatter!(part .+ fitted_model.param[1] * (ui .- u0), vi )    
#     f
# end

ab_vec_data = vectorise(ab_suvr)
tau_vec_data = vectorise(tau_suvr)
vol_vec_data = vectorise(vols)

Random.seed!(1234)

m = ensemble_atn_truncated(prob, inits, ts, ab_tidx, tau_tidx, n_subjects)
pst = m | (ab_data = ab_vec_data, tau_data = tau_vec_data, vol_data = vol_vec_data);
pst()

n_samples = 1000
n_chains = 1
println("Starting Inference")
samples = sample(pst, NUTS(0.8), MCMCSerial(), n_samples, n_chains)
println("Number of Divergences: $(sum(samples[:numerical_error]))")
display(summarize(samples))

using Serialization
serialize(projectdir("output/chains/population-atn/pst-samples-bf-random-$(n_chains)x$(n_samples).jls"), samples)