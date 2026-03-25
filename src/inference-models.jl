module InferenceModels 

using Turing
using LinearAlgebra: I
using ATNModelling.SimulationUtils: _make_atn_prob_func, _atn_output_func, split_sols_ensemble,
                   success_condition, get_retcodes
using DifferentialEquations: ODEProblem, EnsembleProblem, Tsit5, solve

"""
    ensemble_atn_harmonised(fbb_prob, fbb_inits, fbb_times, fbb_ab_tidx, fbb_tau_tidx, fbb_idx, fbb_n,
                            fbp_prob, fbp_inits, fbp_times, fbp_ab_tidx, fbp_tau_tidx, fbp_idx, fbp_n, n)

Hierarhical probabilitstic model for calibrating the ATN with longitudinal neuroimaging data with 
Aβ data from both Florbetaben (fbb) and Florbetapir(fbp). 

This model assumes i.i.d noise for pooled across Aβ tracers and a pooled coupling parameter between Aβ and tau. 
"""
@model function ensemble_atn_harmonised(fbb_prob, fbb_inits, fbb_times, fbb_ab_tidx, fbb_tau_tidx, fbb_idx, fbb_n,
                                        fbp_prob, fbp_inits, fbp_times, fbp_ab_tidx, fbp_tau_tidx, fbp_idx, fbp_n, n)
    σ_a  ~ InverseGamma(2,3)
    σ_t  ~ InverseGamma(2,3)
    σ_v  ~ InverseGamma(2,3)

    Am_a ~ LogNormal() # truncated(Normal(0, 1), lower=0)
    As_a ~ truncated(Normal(0, 1), lower=0)

    Pm_t ~ LogNormal() #truncated(Normal(0, 1), lower=0)
    Ps_t ~ truncated(Normal(0, 1), lower=0)

    Am_t ~ LogNormal() #truncated(Normal(0, 1), lower=0)
    As_t ~ truncated(Normal(0, 1), lower=0)

    Em   ~ LogNormal() #truncated(Normal(0, 1), lower=0)
    Es   ~ truncated(Normal(0, 1), lower=0)

    β_fbb    ~ truncated(Normal(3.21, 3.0), lower=0)
    β_fbp    ~ truncated(Normal(3.68, 3.0), lower=0)

    α_a  ~ filldist(truncated(Normal(Am_a, As_a), lower=0), n)
    ρ_t  ~ filldist(truncated(Normal(Pm_t, Ps_t), lower=0), n)
    α_t  ~ filldist(truncated(Normal(Am_t, As_t), lower=0), n)
    η    ~ filldist(truncated(Normal(Em, Es), lower=0), n)

    fbb_ensemble_prob = EnsembleProblem(fbb_prob, 
                                        prob_func=_make_atn_prob_func(fbb_inits, α_a[fbb_idx], ρ_t[fbb_idx], α_t[fbb_idx], β_fbb, η[fbb_idx], fbb_times), 
                                        output_func=_atn_output_func)

    fbb_esol = solve(fbb_ensemble_prob, Tsit5(), verbose=false, abstol = 1e-6, reltol = 1e-6, trajectories=fbb_n)


    fbp_ensemble_prob = EnsembleProblem(fbp_prob, 
                                        prob_func=_make_atn_prob_func(fbp_inits, α_a[fbp_idx], ρ_t[fbp_idx], α_t[fbp_idx], β_fbp, η[fbp_idx], fbp_times), 
                                        output_func=_atn_output_func)

    fbp_esol = solve(fbp_ensemble_prob, Tsit5(), verbose=false, abstol = 1e-6, reltol = 1e-6, trajectories=fbp_n)

    if !success_condition(get_retcodes(fbb_esol)) || !success_condition(get_retcodes(fbp_esol))
        Turing.@addlogprob! -Inf
        println("failed")
        return nothing
    end

    fbb_preds, fbb_tau_preds, fbb_vol_preds =  split_sols_ensemble(fbb_esol, fbb_ab_tidx, fbb_tau_tidx)

    fbp_preds, fbp_tau_preds, fbp_vol_preds =  split_sols_ensemble(fbp_esol, fbp_ab_tidx, fbp_tau_tidx)

    fbb_data ~ MvNormal(fbb_preds, σ_a^2 * I)
    fbb_tau_data ~ MvNormal(fbb_tau_preds, σ_t^2 * I)
    fbb_vol_data ~ MvNormal(fbb_vol_preds, σ_v^2 * I) 

    fbp_data ~ MvNormal(fbp_preds, σ_a^2 * I)
    fbp_tau_data ~ MvNormal(fbp_tau_preds, σ_t^2 * I)
    fbp_vol_data ~ MvNormal(fbp_vol_preds, σ_v^2 * I) 
end

@model function ensemble_atn(prob, inits, times, ab_tidx, tau_tidx, n)
    σ_a  ~ InverseGamma(2,3)
    σ_t  ~ InverseGamma(2,3)
    σ_v  ~ InverseGamma(2,3)
    
    Am_a ~ LogNormal()
    As_a ~ truncated(Normal(0, 3), lower=0)

    Pm_t ~ LogNormal()
    Ps_t ~ truncated(Normal(0, 3), lower=0)
    
    Am_t ~ LogNormal()
    As_t ~ truncated(Normal(0, 3), lower=0)

    Em   ~ LogNormal()
    Es   ~ truncated(Normal(0, 3), lower=0)
    
    β    ~ truncated(Normal(2.25, 3.), lower=1)
    
    α_a  ~ filldist(truncated(Normal(Am_a, As_a), lower=0), n)
    ρ_t  ~ filldist(truncated(Normal(Pm_t, Ps_t), lower=0), n)
    α_t  ~ filldist(truncated(Normal(Am_t, As_t), lower=0), n)
    η    ~ filldist(truncated(Normal(Em, Es), lower=0), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=make_atn_prob_func(inits, α_a, ρ_t, α_t, β, η, times), 
                                    output_func=atn_output_func)
    
    _esol = solve(ensemble_prob,
                    Tsit5(),
		            verbose=false,
                    abstol = 1e-6, 
                    reltol = 1e-6, 
                    trajectories=n)

    if !success_condition(get_retcodes(_esol))
        Turing.@addlogprob! -Inf
        println("failed")
        return nothing
    end
    ab_preds, tau_preds, vol_preds =  split_sols_ensemble(_esol, ab_tidx, tau_tidx)
    
    ab_data ~ MvNormal(ab_preds, σ_a^2 * I)
    tau_data ~ MvNormal(tau_preds, σ_t^2 * I)
    vol_data ~ MvNormal(vol_preds, σ_v^2 * I) 
end


end