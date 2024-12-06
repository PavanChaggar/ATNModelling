module SimulationUtils

using DifferentialEquations: ODEProblem, ODEFunction, solve, Tsit5, remake
using CSV: read
using DataFrames: DataFrame
using DrWatson: projectdir, datadir
using DelimitedFiles: readdlm

"""
    load_ab_params()

Return a vector of regional baseline valukes and carrying capacities for amyloid.
"""
function load_ab_params()
    ab_params = read(projectdir("output/analysis-derivatives/ab-derivatives/ab-params.csv"), DataFrame)
    return ab_params.u0, ab_params.ui
end

"""
    load_tau_params()

Return a vector of regional baseline valukes, carrying capacities and PART 
capacities for tau.
"""
function load_tau_params()
    tau_params = read(datadir("derivatives/tau-params.csv"), DataFrame)
    sympart = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/pypart-sym.csv")) |> vec
    return tau_params.v0, tau_params.vi, sympart
end

"""
   concentration(v, v0, vi)

Calculate the concentration of `v` between baseline value, `v0` and 
carrying capacity `vi`.
"""
function concentration(v, v0, vi)
    if v0 == vi
        return 0
    else
        (v - v0) / (vi - v0)
    end 
end

"""
   make_atn_model(u0, ui, v0, part, L)

Returns an `ODEFunction` correpsonding to the ATN model with 
fixed parameters `u0`, `ui`, `v0`, `part` and graph Laplacian `L`.
"""
function make_atn_model(u0, ui, v0, part, L)
    function atn(D, x, p, t;)
        u = @view x[1:72]
        v = @view x[73:144]
        a = @view x[145:216]

        α_a, ρ_t, α_t, β, η = p

        _ui = (ui .- u0) #.* (1 .- a)
        _vi = ((part .+ (β .* (u .- u0))) .- v0) #.* ( 1 .- a )
        _vi_max = (part .+ (β .* (ui .- u0)))
        D[1:72] .= α_a .* (u .- u0) .* (_ui .- (u .- u0))
        D[73:144] .= -ρ_t * L * (v .- v0) .+ α_t .* (v .- v0) .* (_vi - (v .- v0))
        D[145:216] .= η .* concentration.(v, v0, _vi_max) .* ( 1 .- a )
        #D[145:216] .= η .* (v .- v0) .* ( 1 .- a )
        return nothing
    end
    return return ODEFunction(atn)
end

"""
    make_prob(model::ODEFunction, inits, tspan, params)

Returns an `ODEProblem` given an `ODEFunction`, initial conditions `inits`, 
time span `tspan`, and parameters `params`. 
"""
function make_prob(model::ODEFunction, inits, tspan, params)
    ODEProblem(model, inits, tspan, params)
end

"""
    simulate(model::ODEFunction, inits, tspan, params; saveat=0.1)

Solve the ODE given by the `ODEFunction` model, with initial conditons `inits`,
parameters `params`, over the time span `tspan`
"""
function simulate(model::ODEFunction, inits, tspan, params; saveat=0.1)
    return solve(make_prob(model, inits, tspan, params), Tsit5(), saveat=saveat)
end

"""
    resimulate(prob::ODEProblem, params; saveat=0.1)

Regenerates an ODEProblem with new parameters `params` and solves the system. 
"""
function resimulate(prob::ODEProblem, params; saveat=0.1)
    _prob = remake(prob, p=params)
    solve(_prob, Tsit5(), saveat=saveat)
end

"""
    generate_data(prob::ODEProblem, t, noise)

Generates data for amyloid, tau and atr given teh ATN model, saved at times 
`t` and with i.i.d Gaussian noise with standard deviation `noise`. 
"""
function generate_data(prob::ODEProblem, t, noise)
    sol = solve(prob, Tsit5(), saveat=t)
    ab = vec(sol[1:72,:]) .+ (randn(size(vec(sol[1:72,:]))) .* noise)
    tau = vec(sol[73:144,:]) .+ (randn(size(vec(sol[73:144,:]))) .* noise)
    atr = vec(sol[145:216,:]) .+ (randn(size(vec(sol[145:216,:]))) .* noise)

    return ab, tau, atr
end

end