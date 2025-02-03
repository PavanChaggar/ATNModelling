module SimulationUtils

using DifferentialEquations: ODEProblem, ODEFunction, solve, Tsit5, remake,
                             EnsembleProblem
using SciMLBase: successful_retcode
using CSV: read
using DataFrames: DataFrame
using DrWatson: projectdir, datadir
using DelimitedFiles: readdlm

"""
    load_ab_params()

Return a vector of regional baseline valukes and carrying capacities for amyloid.
"""
function load_ab_params(;tracer="FBP")
    ab_params = read(projectdir(joinpath("output/analysis-derivatives/ab-derivatives/", tracer, "ab-params.csv")), DataFrame)
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
   conc(v, v0, vi)

Calculate the concentration of `v` between baseline value, `v0` and 
carrying capacity `vi`.
"""
function conc(v, v0, vi)
    if v0 == vi
        return 0
    else
        (v - v0) / (vi - v0)
    end 
end

function make_scaled_atn_model(ui, part, L)
    function atn(D, x, p, t;)
        u = @view x[1:72]
        v = @view x[73:144]
        a = @view x[145:216]

        α_a, ρ_t, α_t, β, η = p
         
        vi = part .+ (β .* u) #.* ( 1 .- a )
        D[1:72] .= α_a .* ui .* u .* (1 .- u) 
        D[73:144] .= -ρ_t * L * v .+ α_t .* vi .* v .* (1 .- v)
        D[145:216] .= η .* v .* ( 1 .- a )
        return nothing
    end
    return ODEFunction(atn)
end

function make_scaled_atn_model_hemisphere(ui, part, L)
    function atn(D, x, p, t;)
        u = @view x[1:36]
        v = @view x[37:72]
        a = @view x[73:108]

        α_a, ρ_t, α_t, β, η = p
         
        vi = part .+ (β .* u) #.* ( 1 .- a )
        D[1:36] .= α_a .* ui .* u .* (1 .- u) 
        D[37:72] .= -ρ_t * L * v .+ α_t .* vi .* v .* (1 .- v)
        D[73:108] .= η .* v .* ( 1 .- a )
        return nothing
    end
    return ODEFunction(atn)
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
        D[145:216] .= η .* conc.(v, v0, _vi_max) .* ( 1 .- a )
        #D[145:216] .= η .* (v .- v0) .* ( 1 .- a )
        return nothing
    end
    return ODEFunction(atn)
end

function make_atn_feedback_model(u0, ui, v0, part, L)
    function atn(D, x, p, t;)
        u = @view x[1:72]
        v = @view x[73:144]
        a = @view x[145:216]

        α_a, ρ_t, α_t, β, η, δ = p
        _ui = (ui .- u0) .* δ .* (1 .- a)
        _vi = ((part .+ (β .* (u .- u0))) .- v0) .* δ .* ( 1 .- a )
        _vi_max = part .+ (β .* _ui)
        D[1:72] .= α_a .* (u .- u0) .* (_ui .- (u .- u0))
        D[73:144] .= -ρ_t * L * (v .- v0) .+ α_t .* (v .- v0) .* (_vi - (v .- v0))
        D[145:216] .= η .* conc.(v, v0, _vi_max) .* ( 1 .- a )
        #D[145:216] .= η .* (v .- v0) .* ( 1 .- a )
        return nothing
    end
    return ODEFunction(atn)
end

function make_atn_fixed_model(u0, ui, v0, L)
    function atn(D, x, p, t;)
        u = @view x[1:72]
        v = @view x[73:144]
        a = @view x[145:216]

        α_a, ρ_t, α_t, κ, β, η = p

        _ui = (ui .- u0) #.* (1 .- a)
        _vi = ((κ .+ (β .* (u .- u0))) .- v0) #.* ( 1 .- a )
        _vi_max = (κ .+ (β .* (ui .- u0)))
        D[1:72] .= α_a .* (u .- u0) .* (_ui .- (u .- u0))
        D[73:144] .= -ρ_t * L * (v .- v0) .+ α_t .* (v .- v0) .* (_vi - (v .- v0))
        D[145:216] .= η .* conc.(v, v0, _vi_max) .* ( 1 .- a )
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

"""
    simulate_amyloid(u::Vector{Float64}, u0::Vector{Float64}, ui::Vector{Float64}, a, t::Float64)

Simulate regional amyloid progression using a logistic model with 
initial conditions `u` that evolves between `u0` and `ui` with rate `a`
and evaluated at time, `t`.
"""
function simulate_amyloid(u::AbstractVector, u0::Vector{Float64}, ui::Vector{Float64}, a, t::Number)
    x = u .- u0
    ((x .* ui .* exp.(ui .* a .* t)) ./ (ui .- x .+ x .* exp.(ui .* a .* t))) .+ u0
end

"""
    simulate_amyloid(u::Vector{Float64}, u0::Vector{Float64}, ui::Vector{Float64}, a, ts::Vector{Float64})

Simulate regional amyloid progression using a logistic model with 
initial conditions `u` that evolves between `u0` and `ui` with rate `a`
and evaluated at times, `ts`.
"""
function simulate_amyloid(u::Vector{Float64}, u0::Vector{Float64}, ui::Vector{Float64}, a, ts::Vector{Float64})
    reduce(hcat, [simulate_amyloid(u, u0, ui, a, t) for t in ts])
end

"""
    simulate_amyloid(us::Vector{Vector{Float64}}, u0::Vector{Float64}, ui::Vector{Float64}, 
                      as, ts::Vector{Vector{Float64}})

Simulate multiple trjacetories of regional amyloid progression 
using a logistic model with initial conditions `us` that evolves 
between `u0` and `ui` with rates `as` and evaluated at times, `ts`.

`us`, `as` and `ts` should have the same length, each corresponding to 
an individual trajectory.
"""
function simulate_amyloid(us::Vector{Vector{Float64}}, u0::Vector{Float64}, ui::Vector{Float64}, 
                          as, ts::Vector{Vector{Float64}})
        [simulate_amyloid(u, u0, ui, a, t) for (u, a, t) in zip(us, as, ts)]
end

function make_atn_prob_func(initial_conditions, α_a, ρ_t, α_t, β, η, _times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], 
                     p=[α_a[i], ρ_t[i], α_t[i], β, η[i]], saveat=_times[i])
    end
end

function make_atn_individial_prob_func(initial_conditions, α_a, ρ_t, α_t, β, η, _times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], 
                     p=[α_a[i], ρ_t[i], α_t[i], β[i], η[i]], saveat=_times[i])
    end
end

function make_atn_fixed_prob_func(initial_conditions, α_a, ρ_t, α_t, κ, β, η, _times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], 
                     p=[α_a[i], ρ_t[i], α_t[i], κ, β, η[i]], saveat=_times[i])
    end
end

function make_atn_feedback_prob_func(initial_conditions, α_a, ρ_t, α_t, β, η, δ, _times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], 
                     p=[α_a[i], ρ_t[i], α_t[i], β, η[i], δ], saveat=_times[i])
    end
end

function atn_output_func(sol,i)
    (sol,false)
end

function split_sols_ensemble(esol, ab_idx, tau_idx)
    d = [[vec(s[1:72, a_idx]), vec(s[73:144, t_idx]), vec(s[145:216, t_idx])] 
          for (s, a_idx, t_idx) in zip(esol, ab_idx, tau_idx)]
    ab = reduce(vcat, [_d[1] for _d in d])
    tau = reduce(vcat, [_d[2] for _d in d])
    vol = reduce(vcat, [_d[3] for _d in d])     
    return ab, tau, vol
end

function split_sols_serial(s, a_idx, t_idx)
    vec(s[1:72, a_idx]), vec(s[73:144, t_idx]), vec(s[145:216, t_idx])
end

function get_retcodes(es)
    [successful_retcode(sol) for sol in es]
end

function success_condition(retcodes)
    allequal(retcodes) && retcodes[1] == 1
end


end