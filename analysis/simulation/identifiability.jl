using ATNModelling.SimulationUtils: load_ab_params, load_tau_params,
                                    make_atn_model, make_prob, simulate, resimulate,
                                    generate_data
using ATNModelling.ConnectomeUtils: get_connectome
using ATNModelling.InferenceModels: atn_inference, fit_model
using Connectomes: laplacian_matrix, get_label
using FileIO
using DrWatson: projectdir
using Serialization
    # --------------------------------------------------------------------------------
    # Simulation set-up
    # --------------------------------------------------------------------------------
u0, ui = load_ab_params()
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2)
L = laplacian_matrix(c)


α_a, ρ_t, α_t, β, η = 0.75, 0.015, 0.5, 3.75, 0.1
params = [α_a, ρ_t, α_t, β, η]

ab_inits = copy(u0)
tau_inits = copy(v0)
atr_inits = zeros(72)

ab_inits .+= 0.2 .* (ui .- u0)

tau_seed_regions = ["entorhinal" ]#,"Left-Amygdala", "Right-Amygdala", "Left-Hippocampus", "Right-Hippocampus"]
tau_seed_idx = findall(x -> get_label(x) ∈ tau_seed_regions, c.parc)
tau_inits[tau_seed_idx] .+= 0.2 .* ((part[tau_seed_idx] .+ β .* (ui[tau_seed_idx] - u0[tau_seed_idx])) .- v0[tau_seed_idx])

inits = [ab_inits; tau_inits; atr_inits]

tspan = (0.0,30.0)
# --------------------------------------------------------------------------------
# Inference study
# --------------------------------------------------------------------------------
f = make_atn_model(u0, ui, v0, part, L)
prob = make_prob(f, inits, tspan, params)

noise = 0.025

ts_slide = [collect(range(0 +i, 3 + i, 3)) for i in 0:3:27]

for t in ts_slide
    _ts = Int.(extrema(t))
    
    _ab, _tau, _atr = generate_data(prob, t, noise)
    save(projectdir("output/synthetic-data/atn-identifiability/pst-$(_ts[1])-$(_ts[end]).jld2"), 
              Dict("ab" => _ab, "tau" => _tau, "atr" => _atr))
    
    _samples = fit_model(atn_inference, _ab, _tau, _atr, prob, t)
    
    serialize(projectdir("output/chains/atn-identifiability/pst-$(_ts[1])-$(_ts[end]).jls"), 
              _samples)
end