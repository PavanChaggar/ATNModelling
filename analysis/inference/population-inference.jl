using ATNModelling.SimulationUtils: make_prob, make_atn_model, 
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: align_data, calc_times, normalise!
using ATNModelling.InferenceModels: fit_model

using Connectomes: laplacian_matrix, get_label
using ADNIDatasets: ADNIDataset, get_id, get_dates, get_initial_conditions, calc_suvr, get_vol, get_times
using DrWatson: projectdir, datadir
using CSV, DataFrames
using SciMLBase: successful_retcode
using DifferentialEquations, Turing, LinearAlgebra
using SciMLSensitivity, ADTypes
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
tau_pos = ADNIDataset(tau_pos_df, dktnames; min_scans=3)

pos_ids = get_id.(tau_pos)
tau_pos_idx = reduce(vcat, [findall(x -> get_id(x) ∈ id, ab_data) for id in pos_ids])

ab_tau_pos = ab_data[tau_pos_idx]
ab_tau_pos_ids = get_id.(ab_tau_pos)

ab_tau_pos_idx = reduce(vcat, [findall(x -> get_id(x) ∈ id, tau_pos) for id in ab_tau_pos_ids])
tau_ab_pos = tau_pos[ab_tau_pos_idx]

ab_times = get_times.(ab_tau_pos)
tau_times = get_times.(tau_ab_pos)

ab_suvr = calc_suvr.(ab_tau_pos)
normalise!(ab_suvr, u0, ui)

tau_suvr = calc_suvr.(tau_ab_pos)
normalise!(tau_suvr, v0)

ab_inits = [ab[:,1] for ab in ab_suvr]
tau_inits = [tau[:,1] for tau in tau_suvr]

delay = [tt[1] - ta[1] for (ta, tt) in zip(calc_times(ab_tau_pos, tau_ab_pos)...)]

tau_pos_vol = get_vol.(tau_ab_pos);
total_vol_norm = [tp ./ sum(tp, dims=1) for tp in tau_pos_vol]
vols = [clamp.(1 .- (vol ./ vol[:,1]), 0, 1) for vol in total_vol_norm]
vol_inits = [vol[:,1] for vol in vols]

function concentration(v, v0, vi)
    if v0 == vi
        return 0
    else
        (v - v0) / (vi - v0)
    end 
end

function tau_atrophy(dx, x, p, t; L = L, u0=u0, ui=ui_diff, v0=v0, part=part)
    v = @view x[1:72]
    a = @view x[73:end]

    u = @view p[1:72]
    α_a, ρ_t, α_t, β, η, d = @view p[73:end]

    vi = (part .+ β .* (simulate_amyloid(u, u0, ui, α_a, t+d) .- u0))
    _vi_max = (part .+ (β .* ui))

    dx[1:72] .= -ρ_t * L * (v .- v0) .+ α_t .* (v .- v0) .* ((vi .- v0) .- (v .- v0))
    dx[73:end] .= η .* concentration.(v, v0, _vi_max) .* ( 1 .- a )
    return nothing
end

inits = [[t; a] for (t, a) in zip(tau_inits, vol_inits)]
n_subjects = length(inits)
tspan = (0., 10.0)
prob = ODEProblem(tau_atrophy, inits[1], tspan, 
                                  [ab_inits[1]; [1.0,1.0,1.0,1.0,1.0,0.0]])

function make_prob_func(initial_conditions, ab_inits, α_a, ρ_t, α_t, β, η, d, times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], 
                        p=[ab_inits[i];  [α_a[i], ρ_t[i], α_t[i], β, η[i], d[i]]], 
                        saveat=times[i])
    end
end
function output_func(sol,i)
    (sol,false)
end

function get_retcodes(es)
    [SciMLBase.successful_retcode(sol) for sol in es]
end

function success_condition(retcodes)
    allequal(retcodes) && retcodes[1] == 1
end

function vec_sol(asol)
    reduce(vcat, reduce.(vcat, asol))
end

function split_sols(esol)
    tau = [s[1:72, :] for s in esol]
    atr = [s[73:end, :] for s in esol]
    return vec_sol(tau), vec_sol(atr)
end

@model function population_atn_inference(ab_inits, inits,
                                        ab_times, tau_times, delays, n)
    σ_a ~ InverseGamma(2,3)
    σ_t ~ InverseGamma(2,3)
    σ_v ~ InverseGamma(2,3)
    
    Pm_t ~ LogNormal() #Beta(2,2) #LogNormal(0.0, 1.0) #Uniform(0.0,3.0)
    Ps_t ~ truncated(Normal(), lower=0)

    Am_a ~ LogNormal()
    As_a ~ truncated(Normal(), lower=0)
    
    Am_t ~ LogNormal()
    As_t ~ truncated(Normal(), lower=0)

    Em ~ LogNormal()
    Es ~ truncated(Normal(), lower=0)
    
    β ~ truncated(Normal(3, 3), lower=0)

    ρ_t ~ filldist(truncated(Normal(Pm_t, Ps_t), lower=0), n)
    α_a ~ filldist(truncated(Normal(Am_a, As_a), lower=0), n)
    α_t ~ filldist(truncated(Normal(Am_t, As_t), lower=0), n)
    η ~ filldist(truncated(Normal(Em, Es), lower=0), n)

    ab_sols = simulate_amyloid(ab_inits, u0, ui, α_a, ab_times)

    tn_ensemble_prob = EnsembleProblem(prob,
                        prob_func=make_prob_func(inits, ab_inits,
                                                 α_a, ρ_t, α_t, β, η, delays, 
                                                tau_times), 
                        output_func=output_func)
    tn_sols = solve(tn_ensemble_prob, Tsit5(), trajectories=n,
                    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                    reltol=1e-6, abstol=1e-6)

    if !success_condition(get_retcodes(tn_sols))
        Turing.@addlogprob! -Inf
        return nothing
    end
    ab_pred = vec_sol(ab_sols)
    tau_pred, vol_pred = split_sols(tn_sols)

    ab_data ~ MvNormal(ab_pred, σ_a^2 * I)
    tau_data ~ MvNormal(tau_pred, σ_t^2 * I)
    vol_data ~ MvNormal(vol_pred, σ_v^2 * I)
    return nothing
end

ab_vec_data = vec_sol(ab_suvr)
tau_vec_data = vec_sol(tau_suvr)
vol_vec_data = vec_sol(vols)
pst_samples = fit_model(population_atn_inference, 
                        ab_vec_data, tau_vec_data, vol_vec_data, 
                        ab_inits, inits, ab_times, tau_times, delay, n_subjects;
                        n_samples = 1000, n_chains=1)

using Serialization
serialize(projectdir("output/chains/population-atn/pst_samples-test.jls"), pst_samples)
