module InferenceModels 

using Turing
using LinearAlgebra: I
using ATNModelling.SimulationUtils: resimulate, simulate_amyloid, 
                   _make_atn_prob_func, _make_atn_feedback_prob_func, _atn_output_func, split_sols_ensemble, split_sols_serial,
                   success_condition, get_retcodes, _make_atn_fixed_prob_func, _make_atn_individial_prob_func
using SciMLBase: successful_retcode
using DifferentialEquations: ODEProblem, EnsembleProblem, Tsit5, solve, remake
using ADTypes: AutoForwardDiff
using SciMLSensitivity: InterpolatingAdjoint, ReverseDiffVJP

"""
    atn_inference(prob::ODEProblem, t)

Generative model of the ATN system at times `t`. The model 
assumes inverse Gamma priors on observation noise and 
half-Normal priors on model parameters. The likelihood function 
assumes i.i.d Gaussian noise for each data modality.
""" 
@model function atn_inference(prob::ODEProblem, t)
    σ_a ~ InverseGamma(2., 3.)
    σ_t ~ InverseGamma(2., 3.)
    σ_v ~ InverseGamma(2., 3.)

    ρ_t ~ truncated(Normal(), lower=0)
    α_a ~ truncated(Normal(), lower=0)
    α_t ~ truncated(Normal(), lower=0)
    β ~ truncated(Normal(0.0, 3.0), lower=0)
    η ~ truncated(Normal(), lower=0)
    
    sol = resimulate(prob, [α_a, ρ_t, α_t, β, η], saveat=t)

    if !successful_retcode(sol)
        Turing.@addlogprob! -Inf
        println("failed")
        return nothing
    end
    
    ab_data ~ MvNormal(vec(sol[1:72,:]), σ_a^2 * I)
    tau_data ~ MvNormal(vec(sol[73:144,:]), σ_t^2 * I)
    vol_data ~ MvNormal(vec(sol[145:216,:]), σ_v^2 * I)
end

"""
    population_ab_inference(inits, u0, ui, ts, n)

Inference model for longitudinal amyloid SUVR with inter-individual 
variability on amyloid production. 
The model assumes i.i.d noise between individuals and a Gaussian 
likelihood function. The noise prior is an InverseGamma(2,3) distribution 
and the ammyloid production parameter is a standard Normal. 
"""
@model function population_ab_inference(inits, u0, ui, ts, n)
    σ ~ InverseGamma(2, 3)

    α_m ~ Normal()
    α_s ~ LogNormal()

    α ~ filldist(Normal(α_m, α_s), n)

    sols = simulate_amyloid(inits, u0, ui, α, ts)
    vecsol = reduce(vcat, reduce.(vcat, sols))
    
    ab_data ~ MvNormal(vecsol, σ^2 * I)  
end

"""
    fit_model(model, ab, tau, atr, args...; n_samples=1000, n_chains=1)

Fits a generative model `model`, to amyloid biomarker data `ab`. 
Args should follow the input order of the `model`. Sampling is performed using 
a NUTS sampler with default settings.
"""
function fit_ab_model(model, ab, args...; n_samples=1000, n_chains=1)
    m = model(args...)
    ab_vec_data =reduce(vcat, reduce.(vcat, ab))
    pst = m | (ab_data = ab_vec_data,);
    samples = sample(pst, NUTS(), MCMCSerial(), n_samples, n_chains)
    println("Number of Divergences: $(sum(samples[:numerical_error]))")
    display(summarize(samples))
    return samples
end

"""
    ensemble_atn(prob::ODEProblem, 
                 inits::Vector{Vector{Float64}}, 
                 times::Vector{Vector{Float64}}, 
                 ab_tidx::Vector{Vector{Int64}}, 
                 tau_tidx::Vector{Vector{Int64}}, 
                 n::Int)

Hierarhical probabilitstic model for calibrating the ATN with longitudinal neuroimaging data. 
The model assumes a pooled coupling parameter between Aβ and tau. 
"""
@model function ensemble_atn(prob::ODEProblem, inits::Vector{Vector{Float64}}, times::Vector{Vector{Float64}}, ab_tidx::Vector{Vector{Int64}}, tau_tidx::Vector{Vector{Int64}}, n::Int)
    σ_a  ~ InverseGamma(2,3)
    σ_t  ~ InverseGamma(2,3)
    σ_v  ~ InverseGamma(2,3)
    
    Am_a ~ LogNormal()
    As_a ~ truncated(Normal(0, 1), lower=0)

    Pm_t ~ LogNormal()
    Ps_t ~ truncated(Normal(0, 1), lower=0)
    
    Am_t ~ LogNormal()
    As_t ~ truncated(Normal(0, 1), lower=0)

    Em   ~ LogNormal()
    Es   ~ truncated(Normal(0, 1), lower=0)
    
    β    ~ truncated(Normal(3.5, 3.0), lower=0.)
    
    α_a  ~ filldist(truncated(Normal(Am_a, As_a), lower=0), n)
    ρ_t  ~ filldist(truncated(Normal(Pm_t, Ps_t), lower=0), n)
    α_t  ~ filldist(truncated(Normal(Am_t, As_t), lower=0), n)
    η    ~ filldist(truncated(Normal(Em, Es), lower=0), n)

    ensemble_prob = EnsembleProblem(prob, 
                                    prob_func=_make_atn_prob_func(inits, α_a, ρ_t, α_t, β, η, times), 
                                    output_func=_atn_output_func)
    
    _esol = solve(ensemble_prob,
                    Tsit5(),
		            verbose=false,
                    abstol = 1e-6, 
                    reltol = 1e-6, 
                    trajectories=n)

    if !success_condition(get_retcodes(_esol))
        Turing.@addlogprob! -Inf
        println(findall(x -> x == 0, get_retcodes(_esol)))
        println("failed")
        return nothing
    end
    ab_preds, tau_preds, vol_preds =  split_sols_ensemble(_esol, ab_tidx, tau_tidx)
    
    ab_data ~ MvNormal(ab_preds, σ_a^2 * I)
    tau_data ~ MvNormal(tau_preds, σ_t^2 * I)
    vol_data ~ MvNormal(vol_preds, σ_v^2 * I) 
end

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
    
    Am_a ~ truncated(Normal(0, 3), lower=0)    # LogNormal()
    As_a ~ truncated(Normal(0, 3), lower=0)

    Pm_t ~ truncated(Normal(0, 3), lower=0)    # LogNormal()
    Ps_t ~ truncated(Normal(0, 3), lower=0)
    
    Am_t ~ truncated(Normal(0, 3), lower=0)    # LogNormal()
    As_t ~ truncated(Normal(0, 3), lower=0)

    Em   ~ truncated(Normal(0, 3), lower=0)    # LogNormal()
    Es   ~ truncated(Normal(0, 3), lower=0)
    
    β    ~ truncated(Normal(3.5, 3.0), lower=0.)
    
    α_a  ~ filldist(truncated(Normal(Am_a, As_a), lower=0), n)
    ρ_t  ~ filldist(truncated(Normal(Pm_t, Ps_t), lower=0), n)
    α_t  ~ filldist(truncated(Normal(Am_t, As_t), lower=0), n)
    η    ~ filldist(truncated(Normal(Em, Es), lower=0), n)

    fbb_ensemble_prob = EnsembleProblem(fbb_prob, 
                                    prob_func=make_atn_prob_func(fbb_inits, α_a[fbb_idx], ρ_t[fbb_idx], α_t[fbb_idx], β, η[fbb_idx], fbb_times), 
                                    output_func=atn_output_func)
    
    fbb_esol = solve(fbb_ensemble_prob, Tsit5(), verbose=false, abstol = 1e-6, reltol = 1e-6, trajectories=fbb_n)


    fbp_ensemble_prob = EnsembleProblem(fbp_prob, 
                                    prob_func=make_atn_prob_func(fbp_inits, α_a[fbp_idx], ρ_t[fbp_idx], α_t[fbp_idx], β, η[fbp_idx], fbp_times), 
                                    output_func=atn_output_func)
    
    fbp_esol = solve(fbp_ensemble_prob, Tsit5(), verbose=false, abstol = 1e-6, reltol = 1e-6, trajectories=fbp_n)

    if !success_condition(get_retcodes(fbb_esol)) && !success_condition(get_retcodes(fbp_esol))
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


"""
    ensemble_atn_harmonised_tracer(fbb_prob, fbb_inits, fbb_times, fbb_ab_tidx, fbb_tau_tidx, fbb_idx, fbb_n,
                                   fbp_prob, fbp_inits, fbp_times, fbp_ab_tidx, fbp_tau_tidx, fbp_idx, fbp_n, n)

Hierarhical probabilitstic model for calibrating the ATN with longitudinal neuroimaging data with 
Aβ data from both Florbetaben (fbb) and Florbetapir(fbp). 

This model assumes i.i.d noise for each Aβ tracers and a pooled coupling parameter between Aβ and tau. 
"""
@model function ensemble_atn_harmonised_tracer(fbb_prob, fbb_inits, fbb_times, fbb_ab_tidx, fbb_tau_tidx, fbb_idx, fbb_n,
                                        fbp_prob, fbp_inits, fbp_times, fbp_ab_tidx, fbp_tau_tidx, fbp_idx, fbp_n, n)
    σ_fbb  ~ InverseGamma(2,3)
    σ_fbp  ~ InverseGamma(2,3)
    σ_t  ~ InverseGamma(2,3)
    σ_v  ~ InverseGamma(2,3)

    Am_a ~ truncated(Normal(0, 1), lower=0)
    As_a ~ truncated(Normal(0, 1), lower=0)

    Pm_t ~ truncated(Normal(0, 1), lower=0)
    Ps_t ~ truncated(Normal(0, 1), lower=0)

    Am_t ~ truncated(Normal(0, 1), lower=0)
    As_t ~ truncated(Normal(0, 1), lower=0)

    Em   ~ truncated(Normal(0, 1), lower=0)
    Es   ~ truncated(Normal(0, 1), lower=0)

    β    ~ truncated(Normal(3.5, 3.0), lower=0.)

    α_a  ~ filldist(truncated(Normal(Am_a, As_a), lower=0), n)
    ρ_t  ~ filldist(truncated(Normal(Pm_t, Ps_t), lower=0), n)
    α_t  ~ filldist(truncated(Normal(Am_t, As_t), lower=0), n)
    η    ~ filldist(truncated(Normal(Em, Es), lower=0), n)

    fbb_ensemble_prob = EnsembleProblem(fbb_prob, 
    prob_func=make_atn_prob_func(fbb_inits, α_a[fbb_idx], ρ_t[fbb_idx], α_t[fbb_idx], β, η[fbb_idx], fbb_times), 
                                 output_func=atn_output_func)

    fbb_esol = solve(fbb_ensemble_prob, Tsit5(), verbose=false, abstol = 1e-6, reltol = 1e-6, trajectories=fbb_n)

    fbp_ensemble_prob = EnsembleProblem(fbp_prob, 
    prob_func=make_atn_prob_func(fbp_inits, α_a[fbp_idx], ρ_t[fbp_idx], α_t[fbp_idx], β, η[fbp_idx], fbp_times), 
                                 output_func=atn_output_func)

    fbp_esol = solve(fbp_ensemble_prob, Tsit5(), verbose=false, abstol = 1e-6, reltol = 1e-6, trajectories=fbp_n)

    if !success_condition(get_retcodes(fbb_esol)) && !success_condition(get_retcodes(fbp_esol))
        Turing.@addlogprob! -Inf
        println("failed")
        return nothing
    end

    fbb_preds, fbb_tau_preds, fbb_vol_preds =  split_sols_ensemble(fbb_esol, fbb_ab_tidx, fbb_tau_tidx)

    fbp_preds, fbp_tau_preds, fbp_vol_preds =  split_sols_ensemble(fbp_esol, fbp_ab_tidx, fbp_tau_tidx)

    fbb_data ~ MvNormal(fbb_preds, σ_fbb^2 * I)
    fbb_tau_data ~ MvNormal(fbb_tau_preds, σ_t^2 * I)
    fbb_vol_data ~ MvNormal(fbb_vol_preds, σ_v^2 * I) 

    fbp_data ~ MvNormal(fbp_preds, σ_fbp^2 * I)
    fbp_tau_data ~ MvNormal(fbp_tau_preds, σ_t^2 * I)
    fbp_vol_data ~ MvNormal(fbp_vol_preds, σ_v^2 * I) 
end


"""
    ensemble_atn_harmonised_tracer(fbb_prob, fbb_inits, fbb_times, fbb_ab_tidx, fbb_tau_tidx, fbb_idx, fbb_n,
                                   fbp_prob, fbp_inits, fbp_times, fbp_ab_tidx, fbp_tau_tidx, fbp_idx, fbp_n, n)

Hierarhical probabilitstic model for calibrating the ATN with longitudinal neuroimaging data with 
Aβ data from both Florbetaben (fbb) and Florbetapir(fbp). 

This model assumes i.i.d noise pooled across Aβ tracers and a personalised coupling parameter between Aβ and tau. 
"""
@model function ensemble_atn_harmonised_individual(fbb_prob, fbb_inits, fbb_times, fbb_ab_tidx, fbb_tau_tidx, fbb_idx, fbb_n,
    fbp_prob, fbp_inits, fbp_times, fbp_ab_tidx, fbp_tau_tidx, fbp_idx, fbp_n, n)
    σ_a  ~ InverseGamma(2,3)
    σ_t  ~ InverseGamma(2,3)
    σ_v  ~ InverseGamma(2,3)

    Am_a ~ LogNormal()
    As_a ~ truncated(Normal(0, 1), lower=0)

    Pm_t ~ LogNormal()
    Ps_t ~ truncated(Normal(0, 1), lower=0)

    Am_t ~ LogNormal()
    As_t ~ truncated(Normal(0, 1), lower=0)

    Em   ~ LogNormal()
    Es   ~ truncated(Normal(0, 1), lower=0)

    Bm   ~ truncated(Normal(3.5, 3.0), lower=0.)
    Bs   ~ truncated(Normal(0, 3), lower=0)

    α_a  ~ filldist(truncated(Normal(Am_a, As_a), lower=0), n)
    ρ_t  ~ filldist(truncated(Normal(Pm_t, Ps_t), lower=0), n)
    α_t  ~ filldist(truncated(Normal(Am_t, As_t), lower=0), n)
    β    ~ filldist(truncated(Normal(Bm, Bs), lower=0), n)
    η    ~ filldist(truncated(Normal(Em, Es), lower=0), n)

    fbb_ensemble_prob = EnsembleProblem(fbb_prob, 
                                        prob_func=make_atn_individial_prob_func(fbb_inits, α_a[fbb_idx], ρ_t[fbb_idx], α_t[fbb_idx], β[fbb_idx], η[fbb_idx], fbb_times), 
                                        output_func=atn_output_func)

    fbb_esol = solve(fbb_ensemble_prob, Tsit5(), verbose=false, abstol = 1e-6, reltol = 1e-6, trajectories=fbb_n)


    fbp_ensemble_prob = EnsembleProblem(fbp_prob, 
                                        prob_func=make_atn_individial_prob_func(fbp_inits, α_a[fbp_idx], ρ_t[fbp_idx], α_t[fbp_idx], β[fbp_idx], η[fbp_idx], fbp_times), 
                                        output_func=atn_output_func)

    fbp_esol = solve(fbp_ensemble_prob, Tsit5(), verbose=false, abstol = 1e-6, reltol = 1e-6, trajectories=fbp_n)

    if !success_condition(get_retcodes(fbb_esol)) && !success_condition(get_retcodes(fbp_esol))
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

"""
    fit_model(model, ab, tau, atr, args...; n_samples=1000)

Fits a generative model `model`, to biomarker data `ab`, `tau` and `atr`. 
Args should follow the input order of the `model`. Sampling is performed using 
a NUTS sampler with default settings.
"""
function fit_model(model, ab, tau, atr, args...; 
                    n_samples=1000, n_chains=1, adbackend=AutoForwardDiff(chunksize=0))
    m = model(args...)
    pst = m | (ab_data = ab, tau_data = tau, vol_data = atr,);
    pst()
    println("Starting Inference")
    samples = sample(pst, NUTS(0.8; adtype=adbackend), 
    MCMCSerial(), n_samples, n_chains)
    println("Number of Divergences: $(sum(samples[:numerical_error]))")
    display(summarize(samples))
    return samples
end

"""
    serial_atn(prob::ODEProblem, 
                 inits::Vector{Vector{Float64}}, 
                 times::Vector{Vector{Float64}}, 
                 ab_tidx::Vector{Vector{Int64}}, 
                 tau_tidx::Vector{Vector{Int64}}, 
                 n::Int)

Identical to the `ensemble_atn` model but does not use parallelism for ODE solving.
"""
@model function serial_atn(ab_data, tau_data, vol_data, prob, inits, times, ab_tidx, tau_tidx, n)
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
    
    β    ~ truncated(Normal(3.5, 1.), lower=0)

    α_a  ~ filldist(truncated(Normal(Am_a, As_a), lower=0), n)
    ρ_t  ~ filldist(truncated(Normal(Pm_t, Ps_t), lower=0), n)
    α_t  ~ filldist(truncated(Normal(Am_t, As_t), lower=0), n)
    η    ~ filldist(truncated(Normal(Em, Es), lower=0), n)

    for i in eachindex(1:n)
        _prob = remake(prob, u0 = inits[i], p = [α_a[i], ρ_t[i], α_t[i], β, η[i]])
        _sol = solve(_prob, Tsit5(), abstol = 1e-6, reltol = 1e-6, saveat=times[i],
        sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
        if !successful_retcode(_sol)
            Turing.@addlogprob! -Inf
            println("failed")
            break
        end
        ab_preds, tau_preds, vol_preds = split_sols_serial(_sol, ab_tidx[i], tau_tidx[i])
        # ab_data[i] ~ MvNormal(ab_preds, σ_a^2 * I)
        # tau_data[i] ~ MvNormal(tau_preds, σ_t^2 * I)
        # vol_data[i] ~ MvNormal(vol_preds, σ_v^2 * I) 
        Turing.@addlogprob! loglikelihood(MvNormal(ab_preds, σ_a^2 * I),  ab_data[i])
        Turing.@addlogprob! loglikelihood(MvNormal(tau_preds, σ_t^2 * I),  tau_data[i])
        Turing.@addlogprob! loglikelihood(MvNormal(vol_preds, σ_v^2 * I),  vol_data[i])
        
    end
end


"""
    fit_serial_model(model, ab, tau, atr, args...; n_samples=1000)

Fits the serial version of the generative model to ATN biomarker data. 
"""

function fit_serial_atn(model, ab_data, tau_data, vol_data, args...; 
                        n_samples=1000, n_chains=1, adbackend=AutoForwardDiff(chunksize=0))
    ab_vec_data = vec.(ab_data)
    tau_vec_data = vec.(tau_data)
    vol_vec_data = vec.(vol_data)

    pst = model(ab_vec_data, tau_vec_data, vol_vec_data, args...);
    pst()
    println("Starting Inference")
    samples = sample(pst, NUTS(;adtype=adbackend), MCMCSerial(), n_samples, n_chains)
    println("Number of Divergences: $(sum(samples[:numerical_error]))")
    display(summarize(samples))
end

end