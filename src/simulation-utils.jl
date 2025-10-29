module SimulationUtils

using DifferentialEquations: ODEProblem, ODEFunction, solve, Tsit5, remake,
                             EnsembleProblem
using SciMLBase: successful_retcode
using CSV: read
using DataFrames: DataFrame
using DrWatson: projectdir, datadir
using DelimitedFiles: readdlm
using Connectomes: Parcellation, get_label, get_node_id, get_hemisphere, get_lobe
using Turing: mean, Chains
"""
    load_ab_params()

Return a vector of regional baseline valukes and carrying capacities for amyloid.
"""
function load_ab_params(;tracer="FBP")
    if tracer == "FBB" || tracer == "FBP"
        ab_params = read(projectdir(joinpath("output/analysis-derivatives/ab-derivatives/", tracer, "ab-params.csv")), DataFrame)
    elseif tracer == "FMM"
        ab_params = read(projectdir("output/analysis-derivatives/bf/ab-derivatives/ab-params.csv"), DataFrame)
    end
    return ab_params.u0, ab_params.ui
end

"""
    load_tau_params()

Return a vector of regional baseline valukes, carrying capacities and PART 
capacities for tau.
"""
function load_tau_params(;tracer="FTP")
    if tracer == "FTP"
        tau_params = read(datadir("adni-derivatives/tau-params.csv"), DataFrame)
        sympart = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/pypart-sym.csv")) |> vec
    elseif tracer == "RO"
        tau_params = read(datadir("bf-derivatives/tau-params.csv"), DataFrame)
        sympart = readdlm(projectdir("output/analysis-derivatives/bf/tau-derivatives/pypart-sym.csv")) |> vec
    end
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

"""
    dose(c, t, t0)

Return an exponential dosing with dose `c` function if `t`` > `t_0`` or 0 if `t`` < `t_0``. 
"""
function dose(c, t, t0)
    if t < t0   
        return 0 
    else
        return c * exp(-mod(t, 1))
    end
end

"""
   make_scaled_atn_model(u0, ui, v0, part, L)

Returns an `ODEFunction` correpsonding to the scaled ATN model with 
fixed parameters ``ui`, `part` and graph Laplacian `L`.
"""
function make_scaled_atn_model(ui, part, L)
    n = length(ui)
    function atn(D, x, p, t;)
        u = @view x[1:n]
        v = @view x[n+1:2n]
        a = @view x[2n+1:3n]

        α_a, ρ_t, α_t, β, η = p
         
        Δ  = (part .+ (β .* ui)) 
        δ = (part .+ (β .* u .* ui))
        # vi = part .+ (β .* u) #.* ( 1 .- a )
        D[1:n] .= α_a .* ui .* u .* (1 .- u)
        D[n+1:2n] .= -ρ_t * L * v .+ α_t .* Δ .* v .* ((δ./Δ) .- v)
        D[2n+1:3n] .= η .* v .* ( 1 .- a )

        # return nothing
    end
    return ODEFunction(atn)
end

function make_scaled_atn_model_fixed(ui, part, L)
    n = length(ui)
    function atn(D, x, p, t;)
        u = @view x[1:n]
        v = @view x[n+1:2n]
        a = @view x[2n+1:3n]

        α_a, ρ_t, α_t, β, η = p
         
        Δ  = (part .+ (β .* ui)) 
        δ = (part .+ (β .* u .* ui))
        # vi = part .+ (β .* u) #.* ( 1 .- a )
        D[1:n] .= α_a .* ui .* u .* (1 .- u)
        D[n+1:2n] .= -ρ_t * L * v .+ α_t .* Δ .* v .* (1 .- v)
        D[2n+1:3n] .= η .* v .* ( 1 .- a )

        # return nothing
    end
    return ODEFunction(atn)
end
function make_scaled_atn_model_hemisphere(ui, part, L)
    function atn(D, x, p, t;)
        u = @view x[1:36]
        v = @view x[37:72]
        a = @view x[73:108]

        α_a, ρ_t, α_t, β, η = p
         
        vi_max = (part .+ (β .* ui)) 
        vi = (part .+ (β .* u .* ui))
        # vi = part .+ (β .* u) #.* ( 1 .- a )
        D[1:36] .= α_a .* ui .* u .* (1 .- u)
        D[37:72] .= -ρ_t * L * v .+ α_t .* vi .* v .* ((vi./vi_max) .- v)
        D[73:108] .= η .* v .* ( 1 .- a )
        return nothing
    end
    return ODEFunction(atn)
end

"""
   make_scaled_atn_pkpd_model(u0, ui, v0, part, L)

Returns an `ODEFunction` correpsonding to the scaled ATN model on a single hemisphere coupled to a PKPD model. 
This model has fixed parameters ``ui`, `part` and graph Laplacian `L` and distance graph Laplacian `Ld`. `m` is 
a mask of regions in which drug enters the brain and `t0` is the initial dosing time.
"""
# function heaviside(t)
#    0.5 * (sign(t) + 1)
# end
function heaviside(t)
   t > 0 ? 1 : 0
end
function make_scaled_atn_pkpd_model(ui, part, L, Ld, m, t0=0)
    function atn_pkpd(D, x, p, t)
        u = @view x[1:36]
        v = @view x[37:72]
        a = @view x[73:108]
        d = @view x[109:144]
        q = @view x[145:end]
        
        α_a, ρ_t, α_t, β, η, ρ_d, α_d, α_c, λ_d = p
         
        Δ  = (part .+ (β .* ui)) 
        δ = (part .+ (β .* q .* ui))

        f = α_a .* ui .* u .* (1 .- u) .- α_d .* d .* u
        D[1:36] .= α_a .* ui .* u .* (1 .- u) .- α_d .* d .* u
        D[37:72] .= -ρ_t * L * v .+ α_t .* Δ .* v .* ((δ./Δ) .- v)
        D[73:108] .= η .* v .* ( 1 .- a )
        D[109:144] .= -ρ_d * Ld * d .+ dose(α_c, t, t0) .* m .- λ_d .* d 
        D[145:end] .= heaviside.(f) .* (α_a .* ui .* q .* (1 .- q) .- α_d .* d .* q)
    end
    return ODEFunction(atn_pkpd)
end

function make_scaled_atn_pkpd_model_tau(ui, part, L, Ld, m, t0=0)
    function atn_pkpd(D, x, p, t)
        u = @view x[1:36]
        v = @view x[37:72]
        a = @view x[73:108]
        d = @view x[109:144]
        q = @view x[145:end]
        
        α_a, ρ_t, α_t, β, η, ρ_d, α_d, α_c, λ_d = p
         
        Δ  = (part .+ (β .* ui)) 
        δ = (part .+ (β .* q.* ui))
        
        f =  α_t .* Δ .* v .* ((δ./Δ) .- v)
        # println(((δ./Δ) .- f)[29])
        D[1:36] .= α_a .* ui .* u .* (1 .- u) .- α_d .* d .* u
        D[37:72] .= -ρ_t * L * v .+ heaviside.((δ./Δ) .- f) .* f
        D[73:108] .= η .* v .* ( 1 .- a )
        D[109:144] .= -ρ_d * Ld * d .+ dose(α_c, t, t0) .* m .- λ_d .* d 
        D[145:end] .= heaviside.(f) .* (α_a .* ui .* q .* (1 .- q) .- α_d .* d .* q)
    end
    return ODEFunction(atn_pkpd)
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


# """
#    make_atn_model(u0, ui, v0, part, L)

# Returns an `ODEFunction` correpsonding to the ATN model with 
# fixed parameters `u0`, `ui`, `v0`, `part` and graph Laplacian `L`.
# """
# function make_atn_pkpd_model(u0, ui, v0, part, L,  Ld, m, t0=0)
#     function atn(D, x, p, t;)
#        u = @view x[1:36]
#         v = @view x[37:72]
#         a = @view x[73:108]
#         d = @view x[109:144]
        
#         α_a, ρ_t, α_t, β, η, ρ_d, α_d, α_c, λ_d = p

#         _ui = (ui .- u0) #.* (1 .- a)
#         _vi = (part .+ (β .* (u .- u0))) #.* ( 1 .- a )
#         p = _vi .- v
#         vi = (heaviside.(-1 .* p) .* v) + (heaviside.(p) .* _vi) #.* ( 1 .- a )
#         _vi_max = (part .+ (β .* (ui .- u0)))
#         D[1:36] .= α_a .* (u .- u0) .* (_ui .- (u .- u0)) .- α_d .* d .* (u .- u0)
#         D[37:72] .= -ρ_t * L * (v .- v0) .+ α_t .* (v .- v0) .* ((vi .- v0) - (v .- v0))
#         D[73:108] .= η .* conc.(v, v0, _vi_max) .* ( 1 .- a )
#         D[109:144] .= -ρ_d * Ld * d .+ dose(α_c, t, t0) .* m .- λ_d .* d 
#         # D[145:end] .=  heaviside.(D[1:36]) .* (α_a .* (q .- u0) .* (_ui .- (q .- u0)))
#         #D[145:216] .= η .* (v .- v0) .* ( 1 .- a )
#         return nothing
#     end
#     return ODEFunction(atn)
# end


"""
   make_atn_model(u0, ui, v0, part, L)

Returns an `ODEFunction` correpsonding to the ATN model with 
fixed parameters `u0`, `ui`, `v0`, `part` and graph Laplacian `L`.
"""
function make_atn_pkpd_model(u0, ui, v0, part, L,  Ld, m, t0=0)
    function atn(D, x, p, t;)
        u = @view x[1:36]
        v = @view x[37:72]
        a = @view x[73:108]
        d = @view x[109:144]
        q = @view x[145:end]
        
        α_a, ρ_t, α_t, β, η, ρ_d, α_d, α_c, λ_d = p

        _ui = (ui .- u0) #.* (1 .- a)
        _vi = ((part .+ (β .* (q .- u0))) .- v0) #.* ( 1 .- a )
        _vi_max = (part .+ (β .* (ui .- u0)))
        D[1:36] .= α_a .* (u .- u0) .* (_ui .- (u .- u0)) .- α_d .* d .* (u .- u0)
        D[37:72] .= -ρ_t * L * (v .- v0) .+ α_t .* (v .- v0) .* (_vi - (v .- v0))
        D[73:108] .= η .* conc.(v, v0, _vi_max) .* ( 1 .- a )
        D[109:144] .= -ρ_d * Ld * d .+ dose(α_c, t, t0) .* m .- λ_d .* d 
        D[145:end] .=  heaviside.(D[1:36]) .* (α_a .* (q .- u0) .* (_ui .- (q .- u0)))
        #D[145:216] .= η .* (v .- v0) .* ( 1 .- a )
        return nothing
    end
    return ODEFunction(atn)
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
function simulate(model::ODEFunction, inits, tspan, params; saveat=0.1, tol=1e-6)
    return solve(make_prob(model, inits, tspan, params), Tsit5(), abstol = tol, reltol=tol, saveat=saveat)
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

function _make_atn_prob_func(initial_conditions, α_a, ρ_t, α_t, β, η, _times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], 
                     p=[α_a[i], ρ_t[i], α_t[i], β, η[i]], saveat=_times[i])
    end
end

function _make_atn_individial_prob_func(initial_conditions, α_a, ρ_t, α_t, β, η, _times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], 
                     p=[α_a[i], ρ_t[i], α_t[i], β[i], η[i]], saveat=_times[i])
    end
end

function _make_atn_fixed_prob_func(initial_conditions, α_a, ρ_t, α_t, κ, β, η, _times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], 
                     p=[α_a[i], ρ_t[i], α_t[i], κ, β, η[i]], saveat=_times[i])
    end
end

function _make_atn_feedback_prob_func(initial_conditions, α_a, ρ_t, α_t, β, η, δ, _times)
    function prob_func(prob,i,repeat)
        remake(prob, u0=initial_conditions[i], 
                     p=[α_a[i], ρ_t[i], α_t[i], β, η[i], δ], saveat=_times[i])
    end
end

function _atn_output_func(sol,i)
    (sol,false)
end

"""
    split_sols_ensemble(esol, ab_idx, tau_idx)

Separate an `EnsembleSolution` for each ATN biomarker. 
"""
function split_sols_ensemble(esol, ab_idx, tau_idx)
    d = [[vec(s[1:72, a_idx]), vec(s[73:144, t_idx]), vec(s[145:216, t_idx])] 
          for (s, a_idx, t_idx) in zip(esol, ab_idx, tau_idx)]
    ab = reduce(vcat, [_d[1] for _d in d])
    tau = reduce(vcat, [_d[2] for _d in d])
    vol = reduce(vcat, [_d[3] for _d in d])     
    return ab, tau, vol
end

"""
    split_sols_serial(esol, ab_idx, tau_idx)

Separate a single `ODESolution` for each ATN biomarker. 
"""
function split_sols_serial(s, a_idx, t_idx)
    vec(s[1:72, a_idx]), vec(s[73:144, t_idx]), vec(s[145:216, t_idx])
end

"""
    get_retcodes(es)

Return retcodes for an `EnsembleSolution`.
"""
function get_retcodes(es)
    [successful_retcode(sol) for sol in es]
end

"""
    success_condition(retcodes)

Confirm all ODE solutions are successful.  
"""

function success_condition(retcodes)
    allequal(retcodes) && retcodes[1] == 1
end

"""
    get_sub_params(pst::Chains, n::UnitRange, k::Int)

Return average parameter for length(n) subjects from from the kth sample of the posterior chain, pst.
"""
function get_sub_params(pst, n, k)
    _p = [mean([pst["α_a[$i]"][k] for i in n]),
            mean([pst["ρ_t[$i]"][k] for i in n]),
            mean([pst["α_t[$i]"][k] for i in n]),
            pst["β_fbb"][k],
            mean([pst["η[$i]"][k] for i in n])]
end
function get_sub_params_fixed_beta(pst, n, k, beta)
    _p = [mean([pst["α_a[$i]"][k] for i in n]),
            mean([pst["ρ_t[$i]"][k] for i in n]),
            mean([pst["α_t[$i]"][k] for i in n]),
            mean([pst["η[$i]"][k] for i in n])]
    return [_p[1], _p[2], _p[3], beta, _p[4]]
end
"""
    calculate_colocalisation_order(parc::Parcellation, pst, model, inits, tau_threshold, ab_threshold)

Returns a `DataFrame` containing the ordered list of regions in `parc` according to their colocalisation order. 

The colocalisation order is determined by simulating a solution to `model` using posterior parameters from `pst` and initial conditions `inits`, 
with a given `tau_threshold` and `ab_threshold`.
"""

function calculate_colocalisation_order(parc::Parcellation, pst::Chains, model, inits, tau_threshold, ab_threshold; tracer)
    meanpst = mean(pst)
    params = meanpst[:Am_a, :mean], meanpst[:Pm_t, :mean], meanpst[:Am_t, :mean], meanpst[tracer, :mean], meanpst[:Em, :mean]
    # params = mean([get_sub_params(pst, 1:18, i) for i in 1:n_samples])
    _calculate_colocalisation_order(parc::Parcellation, params, model, inits, tau_threshold, ab_threshold)
end

function calculate_colocalisation_order(parc::Parcellation, pst::Chains, beta::Float64, model, inits, tau_threshold, ab_threshold)
    meanpst = mean(pst)
    params = meanpst[:Am_a, :mean], meanpst[:Pm_t, :mean], meanpst[:Am_t, :mean], beta, meanpst[:Em, :mean]
    # params = mean([get_sub_params_fixed_beta(pst, 1:18, i, beta) for i in 1:n_samples])

    _calculate_colocalisation_order(parc::Parcellation, params, model, inits, tau_threshold, ab_threshold)
end

function _calculate_colocalisation_order(parc::Parcellation, params, model, inits, tau_threshold::Vector{Float64}, ab_threshold::Vector{Float64})
    nodes = length(parc)

    sol = simulate(model, inits, (0, 200), params, saveat=0.01)

    asol = Array(sol)
    ab_sol = asol[1:nodes,:]
    tau_sol = asol[collect(1:nodes) .+ nodes,:]

    tau_seed_idx = tau_sol .>= tau_threshold
    ab_seed_idx = ab_sol .>= ab_threshold

    ab_tau_coloc = tau_seed_idx .* ab_seed_idx

    ab_tau_coloc_time = Vector{Float64}()
    for i in eachrow(ab_tau_coloc)
        push!(ab_tau_coloc_time, sol.t[findfirst(x -> x == 1, i)])
    end

    tau_time = Vector{Float64}()
    for i in eachrow(tau_seed_idx)
        push!(tau_time, sol.t[findfirst(x -> x == 1, i)])
    end
    
    df = DataFrame(RegionID = collect(1:nodes), 
                   DKTID = get_node_id.(parc), 
                   Region = get_label.(parc), 
                   Hemisphere = get_hemisphere.(parc),
                   Coloc_time = ab_tau_coloc_time,
                   tau_time = tau_time)
    sorted_df = sort(df, :Coloc_time)
    sorted_df.Order = 1:nodes
    return sorted_df
end

function _calculate_colocalisation_order(parc::Parcellation, params, model, inits, tau_threshold, ab_threshold)
    nodes = length(parc)

    sol = simulate(model, inits, (0, 200), params, saveat=0.01)

    asol = Array(sol)
    ab_sol = asol[1:nodes,:]
    tau_sol = asol[collect(1:nodes) .+ nodes,:]

    tau_seed = findall(x -> x >= tau_threshold, tau_sol)
    tau_seed_idx = zeros(nodes, size(asol,2))
    tau_seed_idx[tau_seed] .= 1.0

    ab_seed = findall(x -> x >= ab_threshold, ab_sol)
    ab_seed_idx = zeros(nodes, size(asol,2))
    ab_seed_idx[ab_seed] .= 1.0

    ab_tau_coloc = tau_seed_idx .* ab_seed_idx

    ab_tau_coloc_time = Vector{Float64}()
    for i in eachrow(ab_tau_coloc)
        push!(ab_tau_coloc_time, sol.t[findfirst(x -> x == 1, i)])
    end

    tau_time = Vector{Float64}()
    for i in eachrow(tau_seed_idx)
        push!(tau_time, sol.t[findfirst(x -> x == 1, i)])
    end
    

    df = DataFrame(RegionID = collect(1:nodes), 
                   DKTID = get_node_id.(parc), 
                   Region = get_label.(parc), 
                   Hemisphere = get_hemisphere.(parc),
                   Coloc_time = ab_tau_coloc_time,
                   tau_time = tau_time)
    sorted_df = sort(df, :Coloc_time)
    sorted_df.Order = 1:nodes
    return sorted_df
end

"""
    find_seed(parc, sols, tau_threshold, ab_threshold)

For a set of ODE solutions, find the seed colocalisation points given a `tau_threshold` and `ab_threshold`.
"""
function find_seed(parc, sols, tau_threshold, ab_threshold)

    nodes = length(parc)
    seeds = Vector{Vector{Int64}}()
    for sol in sols
        asol = Array(sol)
        ab_sol = asol[1:nodes,:]
        tau_sol = asol[collect(1:nodes) .+ nodes,:]

        tau_seed = findall(x -> x >= tau_threshold, tau_sol)
        tau_seed_idx = zeros(nodes, size(asol,2))
        tau_seed_idx[tau_seed] .= 1.0

        ab_seed = findall(x -> x >= ab_threshold, ab_sol)
        ab_seed_idx = zeros(nodes, size(asol,2))
        ab_seed_idx[ab_seed] .= 1.0

        
        ab_tau_coloc = tau_seed_idx .* ab_seed_idx
        init_idx = findfirst(x -> x > 0, sum(ab_tau_coloc, dims=1))
        if init_idx isa Nothing
            continue
        end
        push!(seeds, findall(x -> x == 1, ab_tau_coloc[:,init_idx[2]]))
    end
    return seeds
end
function find_seed(parc, sols, tau_threshold::Vector{Float64}, ab_threshold::Vector{Float64})

    nodes = length(parc)
    seeds = Vector{Vector{Int64}}()
    for sol in sols
        asol = Array(sol)
        ab_sol = asol[1:nodes,:]
        tau_sol = asol[collect(1:nodes) .+ nodes,:]

         tau_seed_idx = tau_sol .>= tau_threshold
        ab_seed_idx = ab_sol .>= ab_threshold

        ab_tau_coloc = tau_seed_idx .* ab_seed_idx
        init_idx = findfirst(x -> x > 0, sum(ab_tau_coloc, dims=1))
        if init_idx isa Nothing
            continue
        end
        push!(seeds, findall(x -> x == 1, ab_tau_coloc[:,init_idx[2]]))
    end
    return seeds
end


"""
    calculate_colocalisation_prob(parc, pst, model, inits, tau_prob, ab_prob)

For a given model and parcellation, find the initial colocalisation probability given a 
posteriod distribution `pst`, initial conditions `inits`, tau threshold `tau_threshold` and ab threshold `ab_threshold`, 
"""
function calculate_colocalisation_prob(parc, pst, model, inits, tau_threshold, ab_threshold; tracer=:β_fbb)
    sols = [simulate(model, inits, (0, 50), params, saveat=0.05) 
    for params in zip( vec(pst[:Am_a]), vec(pst[:Pm_t]), vec(pst[:Am_t]), vec(pst[tracer]), vec(pst[:Em]))];
    # for params in [get_sub_params(pst, 1:18, i) for i in 1:1000]]
                        
    seed_idx = reduce(vcat, find_seed(parc, sols, tau_threshold, ab_threshold))

    nodes = length(parc)
    seed_count = zeros(nodes)
    seed_count[unique(seed_idx)] .= [count(==(i), seed_idx) for i in unique(seed_idx)]
    seed_prob = seed_count ./ length(seed_idx)
    init_seed_idx = findall(x -> x > 0, seed_prob)

    seed_prob[init_seed_idx]

   df =  DataFrame(DKTID = get_node_id.(parc), 
                   Seed = get_label.(parc), 
                   Hemisphere = get_hemisphere.(parc), 
                   Seed_prob = seed_prob, 
                   lobe=get_lobe.(parc))
    
    return sort(df, :Seed_prob)
end

function calculate_colocalisation_prob(parc, pst, beta, model, inits, tau_threshold, ab_threshold)
sols = [simulate(model, inits, (0, 200), params, saveat=0.1) 
    for params in zip( vec(pst[:Am_a]), vec(pst[:Pm_t]), vec(pst[:Am_t]), beta, vec(pst[:Em]))];
# for params in [get_sub_params_fixed_beta(pst, 1:18, i, beta) for i in 1:1000]]
    seed_idx = reduce(vcat, find_seed(parc, sols, tau_threshold, ab_threshold))

    nodes = length(parc)
    seed_count = zeros(nodes)
    seed_count[unique(seed_idx)] .= [count(==(i), seed_idx) for i in unique(seed_idx)]
    seed_prob = seed_count ./ length(seed_idx)
    init_seed_idx = findall(x -> x > 0, seed_prob)

    seed_prob[init_seed_idx]

   df =  DataFrame(DKTID = get_node_id.(parc), 
                   Seed = get_label.(parc), 
                   Hemisphere = get_hemisphere.(parc), 
                   Seed_prob = seed_prob, 
                   lobe=get_lobe.(parc))
    
    return sort(df, :Seed_prob)
end
end