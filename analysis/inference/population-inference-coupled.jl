using ATNModelling.SimulationUtils: make_prob, make_atn_model, 
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx
using ATNModelling.InferenceModels: fit_model

using Connectomes: laplacian_matrix, get_label
using ADNIDatasets: ADNIDataset, get_id, get_dates, get_initial_conditions, calc_suvr, get_vol, get_times
using DrWatson: projectdir, datadir
using CSV, DataFrames
using SciMLBase: successful_retcode
using DifferentialEquations, Turing, LinearAlgebra
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

atn_model = make_atn_model(u0, ui, v0, part, L)
prob = make_prob(atn_model, 
          [ab_inits[1]; tau_inits[1]; vol_inits[1]], 
          (0.0,7.5), [1.0,1.0,1.0,3.5,1.0])

# function make_prob_func(initial_conditions, ρ_t, α_a, α_t, β, η, _times)
#     function prob_func(prob,i,repeat)
#         remake(prob, u0=initial_conditions[i], 
#                      p=[ρ_t[i], α_a[i], α_t[i], β, η[i]], saveat=_times[i])
#     end
# end
# function output_func(sol,i)
#     (sol,false)
# end

# function split_sols(esol, ab_idx, tau_idx)
#     d = [[vec(s[1:72, a_idx]), vec(s[73:144, t_idx]), vec(s[145:216, t_idx])] 
#           for (s, a_idx, t_idx) in zip(esol, ab_idx, tau_idx)]
#     ab = reduce(vcat, [_d[1] for _d in d])
#     tau = reduce(vcat, [_d[2] for _d in d])
#     vol = reduce(vcat, [_d[3] for _d in d])     
#     return ab, tau, vol
# end

inits = [[ab; tau; vol] for (ab, tau, vol) in zip(ab_inits, tau_inits, vol_inits)]
n_subjects = length(ab)
# ensemble_prob = EnsembleProblem(prob,
#                                     prob_func=make_prob_func(inits, 
#                                     fill(fill(1.0, n_subjects), 3)..., 5.0, ones(n_subjects), ts), 
#                                     output_func=output_func)
# esol = solve(ensemble_prob, Tsit5(), trajectories=n_subjects)

# function get_retcodes(es)
#     [SciMLBase.successful_retcode(sol) for sol in es]
# end

# function success_condition(retcodes)
#     allequal(retcodes) && retcodes[1] == 1
# end

function split_sols_2(s, a_idx, t_idx)
    vec(s[1:72, a_idx]), vec(s[73:144, t_idx]), vec(s[145:216, t_idx])
end

@model function ensemble_fit(ab_data, tau_data, vol_data, prob, inits, times, ab_tidx, tau_tidx, n)
    σ_a  ~ InverseGamma(2,3)
    σ_t  ~ InverseGamma(2,3)
    σ_v  ~ InverseGamma(2,3)
    
    Am_a ~ truncated(Normal(), lower=0)
    As_a ~ truncated(Normal(), lower=0)

    Pm_t ~ truncated(Normal(), lower=0)
    Ps_t ~ truncated(Normal(), lower=0)
    
    Am_t ~ truncated(Normal(), lower=0)
    As_t ~ truncated(Normal(), lower=0)

    Em   ~ truncated(Normal(), lower=0)
    Es   ~ truncated(Normal(), lower=0)
    
    β    ~ truncated(Normal(3, 1), lower=0)

    α_a  ~ filldist(truncated(Normal(Am_a, As_a), lower=0), n)
    ρ_t  ~ filldist(truncated(Normal(Pm_t, Ps_t), lower=0), n)
    α_t  ~ filldist(truncated(Normal(Am_t, As_t), lower=0), n)
    η    ~ filldist(truncated(Normal(Em, Es), lower=0), n)

    for i in eachindex(1:n)
        _prob = remake(prob, u0 = inits[i], p = [α_a[i], ρ_t[i], α_t[i], β, η[i]])
        _sol = solve(_prob, Tsit5(), abstol = 1e-9, reltol = 1e-9, saveat=times[i])
        if !successful_retcode(_sol)
            Turing.@addlogprob! -Inf
            println("failed")
            break
        end
        ab_preds, tau_preds, vol_preds = split_sols_2(_sol, ab_tidx[i], tau_tidx[i])
        # ab_data[i] ~ MvNormal(ab_preds, σ_a^2 * I)
        # tau_data[i] ~ MvNormal(tau_preds, σ_t^2 * I)
        # vol_data[i] ~ MvNormal(vol_preds, σ_v^2 * I) 
        Turing.@addlogprob! loglikelihood(MvNormal(ab_preds, σ_a^2 * I),  ab_data[i])
        Turing.@addlogprob! loglikelihood(MvNormal(tau_preds, σ_t^2 * I),  tau_data[i])
        Turing.@addlogprob! loglikelihood(MvNormal(vol_preds, σ_v^2 * I),  vol_data[i])
        
    end
    # ensemble_prob = EnsembleProblem(prob, 
    #                                 prob_func=make_prob_func(inits, ρ_t, α_a, α_t, β, η, times), 
    #                                 output_func=output_func)
    
    # _esol = solve(ensemble_prob,
    #                 Tsit5(),
	# 	            verbose=false,
    #                 abstol = 1e-9, 
    #                 reltol = 1e-9, 
    #                 trajectories=n, 
    #                 sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

    # if !success_condition(get_retcodes(_esol))
    #     Turing.@addlogprob! -Inf
    #     println(findall(x -> x == 0, get_retcodes(_esol)))
    #     println("failed")
    #     return nothing
    # end
    # ab_preds, tau_preds, vol_preds =  split_sols(_esol, ab_tidx, tau_tidx)
    
    # ab_data ~ MvNormal(ab_preds, σ_a^2 * I)
    # tau_data ~ MvNormal(tau_preds, σ_a^2 * I)
    # vol_data ~ MvNormal(vol_preds, σ_a^2 * I) 
end

# ab_vec_data = reduce(vcat, vec.(ab_suvr))
# tau_vec_data = reduce(vcat, vec.(tau_suvr))
# vol_vec_data = reduce(vcat, vec.(vols))

# m = ensemble_fit(prob, inits, ts, ab_tidx, tau_tidx, n_subjects);
# m()

# pst = m | (ab_data = ab_vec_data, tau_data = tau_vec_data, vol_data = vol_vec_data,)

# pst()

# turing_suite = make_turing_suite(pst; adbackends=[AutoForwardDiff(chunksize=300)])
# run(turing_suite)


# @code_warntype pst.f(
#     pst,
#     Turing.VarInfo(pst),
#     Turing.SamplingContext(
#         Random.GLOBAL_RNG, Turing.SampleFromPrior(), Turing.DefaultContext(),
#     ),
#     pst.args...,
# )

ab_vec_data = vec.(ab_suvr)
tau_vec_data = vec.(tau_suvr)
vol_vec_data = vec.(vols)

m = ensemble_fit(ab_vec_data, tau_vec_data, vol_vec_data, prob, inits, ts, ab_tidx, tau_tidx, n_subjects);
m()

using TuringBenchmarking

pst = sample(m, NUTS(), 1000)

using Serialization
serialize(projectdir("output/chains/population-atn/pst-samples-test-truncated-normal.jls"), pst)

println("Number of Divergences: $(sum(pst[:numerical_error]))")
display(summarize(pst))