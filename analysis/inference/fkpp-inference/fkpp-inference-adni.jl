using ATNModelling.SimulationUtils: make_prob, make_atn_model, 
                                    load_ab_params, load_tau_params, get_retcodes, success_condition
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise, 
                              make_ucsf_df, add_icv
using ATNModelling.InferenceModels: ensemble_atn_harmonised

using Connectomes: laplacian_matrix, get_label
using ADNIDatasets: ADNIDataset, get_id, calc_suvr, get_vol, get_times
using DrWatson: projectdir, datadir
using CSV, DataFrames
using DifferentialEquations, Turing, LinearAlgebra
using Random
using AdvancedHMC
using SciMLSensitivity
using Serialization
# --------------------------------------------------------------------------------
# Script params 
# --------------------------------------------------------------------------------
n_samples = parse(Int, ARGS[1])
n_chains = parse(Int, ARGS[2])
# --------------------------------------------------------------------------------
# Tracer independent data
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c) 
dktnames = get_parcellation() |> get_cortex |> get_dkt_names

# Amyloid data 
_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=3)

ucsf_df = CSV.read(datadir("ADNI/2025/adni-ucsf-crosssec-mri.csv"), DataFrame)
data_dict_df = CSV.read(datadir("ADNI/adni-data-dictionary.csv"), DataFrame)
ucsf_data_dict = filter(x -> x.CRFNAME == "Cross-Sectional FreeSurfer (7.x)", data_dict_df)
atr_df = make_ucsf_df(ucsf_df, ucsf_data_dict, dktnames)
filter!(x -> x.OVERALLQC != "Fail", atr_df)
tau_atr_df = add_icv(tau_pos_df, atr_df; dt_threshold=180)
tau_icv_df = filter(x -> x.Has_ICV, tau_atr_df)

tau_data = ADNIDataset(tau_icv_df, dktnames; min_scans=3)
# --------------------------------------------------------------------------------
# Load fbb data
# --------------------------------------------------------------------------------
tracer="FBB"
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

fbb, fbb_tau = align_data(fbb_data, tau_data; min_tau_scans=3)
@assert get_id.(fbb) == get_id.(fbb_tau)


fbb_tau_times = get_times.(fbb_tau)

fbb_tau_suvr = calc_suvr.(fbb_tau)
normalise!(fbb_tau_suvr, v0)
fbb_tau_inits = [d[:,1] for d in fbb_tau_suvr]

# --------------------------------------------------------------------------------
# Load fbp data
# --------------------------------------------------------------------------------
tracer="FBP"
fbp_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
fbp_data = ADNIDataset(fbp_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

fbp, fbp_tau = align_data(fbp_data, tau_data; min_tau_scans=3)

@assert allequal([x ∉ get_id.(fbp_tau) for x in get_id.(fbb_tau)])
@assert get_id.(fbp) == get_id.(fbp_tau)

fbp_tau_times = get_times.(fbp_tau)

fbp_tau_suvr = calc_suvr.(fbp_tau)
normalise!(fbp_tau_suvr, v0, vi)
fbp_tau_inits = [d[:,1] for d in fbp_tau_suvr]

# --------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------
function NetworkLocalFKPP(dv, v, p, t; L = L, v0 = v0, vi = vi)
    dv .= -p[1] * L * (v .- v0) .+ p[2] .* (v .- v0) .* ((vi .- v0) .- (v .- v0))
end

function make_prob_func(initial_conditions, p, a, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], p=[p[i], a[i]], saveat=times[i])
    end
end

function output_func(sol,i)
    (sol,false)
end

tau_suvr = [fbb_tau_suvr; fbp_tau_suvr]
initial_conditions = [fbb_tau_inits; fbp_tau_inits]
times = [fbb_tau_times; fbp_tau_times]

tau_vec_data = vectorise(tau_suvr)

n_pos = length(initial_conditions)

prob = ODEProblem(NetworkLocalFKPP, 
                  initial_conditions[1], 
                  (0.,maximum(reduce(vcat, times))), 
                  [1.0,1.0])
                  
sol = solve(prob, Tsit5())

ensemble_prob = EnsembleProblem(prob, prob_func=make_prob_func(initial_conditions, ones(n_pos), ones(n_pos), times), output_func=output_func)
ensemble_sol = solve(ensemble_prob, Tsit5(), trajectories=n_pos)

function vec_sol(es)
    reduce(vcat, [vec(sol) for sol in es])
end

# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

@model function localfkpp(prob, initial_conditions, times, n)
    σ ~ InverseGamma(2,3)

    Pm ~ LogNormal(0.0, 1.0)
    Ps ~ truncated(Normal(0, 1), lower=0)

    Am ~ LogNormal(0.0, 1.0)
    As ~ truncated(Normal(0, 1), lower=0)

    ρ ~ filldist(truncated(Normal(Pm, Ps), lower=0), n)
    α ~ filldist(truncated(Normal(Am, As), lower=0), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_prob_func(initial_conditions, ρ, α, times), 
                                    output_func=output_func)

    ensemble_sol = solve(ensemble_prob, 
                         Tsit5(), 
                         abstol = 1e-6, 
                         reltol = 1e-6, 
                         trajectories=n)
    
    if !success_condition(get_retcodes(ensemble_sol))
        Turing.@addlogprob! -Inf
        println("failed")
        return nothing
    end
    vecsol = vec_sol(ensemble_sol)

    data ~ MvNormal(vecsol, σ^2 * I)
end

# setadbackend(:zygote)
Random.seed!(1234)  

m = localfkpp(prob, initial_conditions, times, n_pos);
m();
pst = m | (data = tau_vec_data,)
pst()
println("starting inference")
sample(pst, Prior(), 1000)
pst = sample(m,
             Turing.NUTS(0.8), #, metricT=AdvancedHMC.DenseEuclideanMetric), 
             MCMCSerial(), 
             n_samples, 
             n_chains)
serialize(projectdir("output/chains/local-fkpp/pst-taupos-adni-$(n_chains)x$(n_samples).jls"), pst)