module InferenceModels 

using Turing
using LinearAlgebra: I
using ATNModelling.SimulationUtils: resimulate
using SciMLBase: successful_retcode
using DifferentialEquations: ODEProblem

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
    fit_model(model, ab, tau, atr, args...; n_samples=1000)

Fits a generative model `model`, to biomarker data `ab`, `tau` and `atr`. 
Args should follow the input order of the `model`. Sampling is performed using 
a NUTS sampler with default settings.
"""
function fit_model(model, ab, tau, atr, args...; n_samples=1000)
    m = model(args...)
    pst = m | (ab_data = ab, tau_data = tau, vol_data = atr,);
    samples = sample(pst, NUTS(), n_samples)
    println("Number of Divergences: $(sum(samples[:numerical_error]))")
    display(summarize(samples))
    return samples
end

end