using ATNModelling.SimulationUtils: make_prob, make_atn_model, 
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params, make_atn_fixed_model
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
# Script params 
# --------------------------------------------------------------------------------
tracer = ARGS[1]
n_samples = parse(Int, ARGS[2])
n_chains = parse(Int, ARGS[3])
# --------------------------------------------------------------------------------
# Load parameters
# --------------------------------------------------------------------------------
u0, ui = load_ab_params(tracer=tracer)
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

diagnostics = CSV.read(datadir("ADNI/adni_merge.csv"), DataFrame)

dropmissing!(_ab_data_df, [:AMYLOID_STATUS, :CENTILOIDS])
abpos = filter(x -> x.AMYLOID_STATUS == 1 && x.TRACER == "FBP" && x.CENTILOIDS >= 21, _ab_data_df)
ids = unique(abpos.RID)

idx = [findfirst(isequal(id), abpos.RID) for id in ids]
bl_ab = abpos[idx, :]

diag_idx = filter(!isequal(nothing), [findfirst(isequal(id), diagnostics.RID) for id in ids])
diag_bl = diagnostics[diag_idx,:]

preclinical = filter(x -> x.DX_bl ∈ ["CN", "SMC", "EMCI", "LMCI", "AD"], diag_bl)

preclinical_ab = filter(x -> x.RID ∈ preclinical.RID, abpos)
mean(preclinical_ab.CENTILOIDS)

preclinical_tau = filter(x -> x.RID ∈ preclinical.RID, _tau_data_df)

_ab = ADNIDataset(preclinical_ab, dktnames; min_scans=2, qc=true, reference_region="WHOLECEREBELLUM")
_tau = ADNIDataset(preclinical_tau, dktnames; min_scans=3, qc=true)

ab, tau = align_data(_ab, _tau)

ab_times = get_times.(ab)
tau_times = get_times.(tau)
ts = [sort(unique([a; t])) for (a, t) in zip(ab_times, tau_times)]

ab_tidx = get_time_idx(ab_times, ts)
tau_tidx = get_time_idx(tau_times, ts)

@assert allequal([allequal(ab_times[i] .== ts[i][ab_tidx[i]]) for i in 1:54])
@assert allequal([allequal(tau_times[i] .== ts[i][tau_tidx[i]]) for i in 1:54])

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
          (0.0,7.5), [1.0,1.0,1.0,1.0, 3.5,1.0])
solve(prob, Tsit5())
inits = [[ab; tau; vol] for (ab, tau, vol) in zip(ab_inits, tau_inits, vol_inits)]
n_subjects = length(ab)

# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
using LsqFit

linearmodel(x, p) = part .+ p[1] .* x
fitted_model = curve_fit(linearmodel, ui .- u0, vi, [1.0])
println("params = $(fitted_model.param)")

ab_vec_data = vectorise(ab_suvr)
tau_vec_data = vectorise(tau_suvr)
vol_vec_data = vectorise(vols)

Random.seed!(1234)

m = ensemble_atn(prob, inits, ts, ab_tidx, tau_tidx, n_subjects)
pst = m | (ab_data = ab_vec_data, tau_data = tau_vec_data, vol_data = vol_vec_data,
           β=fitted_model.param[1],);
pst()

println("Starting Inference")
samples = sample(pst, NUTS(0.9), MCMCSerial(), n_samples, n_chains)
println("Number of Divergences: $(sum(samples[:numerical_error]))")
display(summarize(samples))

using Serialization
serialize(projectdir("output/chains/population-atn/pst-samples-fixed-beta-diag-$(tracer)-$(n_chains)x$(n_samples).jls"), samples)