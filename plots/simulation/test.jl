using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, simulate, make_atn_model,
                                    load_ab_params, load_tau_params, conc, make_scaled_atn_model_hemisphere, 
                                    calculate_colocalisation_order, calculate_colocalisation_prob, find_seed
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names,
                                    get_braak_regions

using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise

using Connectomes: get_hemisphere, plot_roi!, get_node_id, get_lobe, laplacian_matrix, get_label
using Colors, ColorSchemes, GLMakie
using DifferentialEquations
using CSV, DataFrames, DrWatson
using ADNIDatasets
using Statistics
using Serialization
# --------------------------------------------------------------------------------
# Connectome and Data
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c) 
cortex = get_parcellation() |> get_cortex 
dktnames = get_dkt_names(cortex)
right_cortex = filter(x -> get_hemisphere(x) == "right", cortex)
right_cortex = filter(x -> get_hemisphere(x) == "left", cortex)
# Amyloid data 
_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 

tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer && x.AMYLOID_STATUS_COMPOSITE_REF == 1, _ab_data_df);
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF")

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 0 && x.NEO_Status == 0, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=1)

pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-fixed-beta-lognormal-1x1000.jls"));
meanpst = mean(pst)
# --------------------------------------------------------------------------------
# Amyloid data
# --------------------------------------------------------------------------------
ab_suvr = calc_suvr.(fbb_data)
normalise!(ab_suvr, fbb_u0, fbb_ui)
ab_conc = map(x -> conc.(x, fbb_u0, fbb_ui), ab_suvr)
ab_inits = [d[:,1] for d in ab_conc]

_mean_ab_init = mean(ab_inits)
_mean_ab_init_sym = (_mean_ab_init[1:36] .+ _mean_ab_init[37:end]) ./ 2
mean_ab_init = [_mean_ab_init_sym; _mean_ab_init_sym]
# mean_ab_init = _mean_ab_init

max_norm(c) =  c ./ maximum(c);
# --------------------------------------------------------------------------------
# Tau data
# --------------------------------------------------------------------------------
tau_suvr = calc_suvr.(tau_data)
vi = part .+ (3.2258211441306877 .* (fbb_ui .- fbb_u0))
normalise!(tau_suvr, v0, vi)
tau_conc = map(x -> conc.(x, v0, vi), tau_suvr)
tau_inits = [d[:,1] for d in tau_conc]

_mean_tau_init = mean(tau_inits)
_mean_tau_init_sym = (_mean_tau_init[1:36] .+ _mean_tau_init[37:end]) ./ 2
mean_tau_init = [_mean_tau_init_sym; _mean_tau_init_sym]
# mean_tau_init = _mean_tau_init
filtered_tau_idx = findall(x -> x < 0.03, mean_tau_init)
mean_tau_init[filtered_tau_idx] .= 0

using CairoMakie; CairoMakie.activate!()
scatter(mean_tau_init[1:36])

vi = part .+ (coupling .* (fbb_ui .- fbb_u0))
Ld = inv(diagm(vi .- v0)) * L * diagm(vi .- v0)

atn_model = make_scaled_atn_model((fbb_ui .- fbb_u0), (part .- v0), Ld)

amyloid_production = 0.34 
tau_transport = 0.05 
tau_production = 0.12
coupling = 4.7851
atrophy = 0.14 
p = [amyloid_production, tau_transport, tau_production, coupling, atrophy]
atn_model = make_scaled_atn_model((fbb_ui .- fbb_u0), (part .- v0), Ld)

prob = ODEProblem(atn_model, [mean_ab_init; mean_tau_init; zeros(72)], (0, 80), p)
sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9)

d1 = reduce(hcat, [sol(t, Val{1}) for t in 0:1:80])
d2 = reduce(hcat, [sol(t, Val{2}) for t in 0:1:80])
d3 = reduce(hcat, [sol(t, Val{3}) for t in 0:1:80])
ab_threshold = Vector{Float64}()
tau_threshold = Vector{Float64}()
for i in 1:72
    ab_dmax = argmax(d1[i, :])
    tau_dmax = argmax(d1[72 + i, :])

    ab_t = argmax(d3[i, ab_dmax:end])
    tau_t = argmax(d3[72 + i, 1:tau_dmax])
    push!(ab_threshold, sol(ab_dmax)[i])
    push!(tau_threshold, sol(tau_t)[72 + i])

end

begin
    cols = Makie.wong_colors()
    CairoMakie.activate!()
    f = Figure(size=(600, 500))
    ax = Axis(f[1,1])
    # plot!(sol, idxs=29, color=cols[1])
    plot!(sol, idxs=72 + 35, color=cols[2])
    ax = Axis(f[2,1])
    for i in [35]
        scatter!(0:1:80, d3[72 + i,:])    
    end
    f
end

ts = [argmax(d) for d in eachrow(d2[73:108,1:50])]
ts_scaled = (ts .- minimum(ts)) ./ (maximum(ts) .- minimum(ts)) 
scatter(ts_scaled)
dktnames[sortperm(ts_scaled)]

using GLMakie, Colors, ColorSchemes; GLMakie.activate!()
using Connectomes
plot_roi(get_node_id.(right_cortex)[sortperm(ts_scaled)], collect(range(0, 1, 36)), ColorSchemes.RdYlBu)

function atn(x, p, ui, part, L; n = 72)
    u = @view x[1:n]
    v = @view x[n+1:2n]
    a = @view x[2n+1:3n]

    α_a, ρ_t, α_t, β, η = p
        
    Δ  = (part .+ (β .* ui)) 
    δ = (part .+ (β .* u .* ui))
    # vi = part .+ (β .* u) #.* ( 1 .- a )
    du = α_a .* ui .* u .* (1 .- u)
    dv = -ρ_t * L * v .+ α_t .* Δ .* v .* ((δ./Δ) .- v)
    da = η .* v .* ( 1 .- a )

    return du, dv, da

end

amyloid_production = 0.34 
tau_transport = 0.05 
tau_production = 0.11
coupling = 4.5
atrophy = 0.15 
p = [amyloid_production, tau_transport, tau_production, coupling, atrophy]
d = [atn(ones(72*3) .* i, p, 
                        fbb_ui .- fbb_u0, part .- v0, Ld) for i in 0:0.01:1]
begin
    f = Figure()
    ax = Axis(f[1,1])
    for (i, t) in enumerate(0:0.01:1)
        scatter!(t, d[i][1][27], color=:grey)
        scatter!(t, d[i][2][27], color=:red)
    end
    f
end

function atn_fixed(D, x, p, t;ui = fbb_ui .- fbb_u0, part = part.-v0, L = Ld)
    u = @view x[1:72]
    v = @view x[73:144]
    a = @view x[145:216]

    α_a, ρ_t, α_t, β, η = p
        
    Δ  = (part .+ (β .* ui)) 
    δ = (part .+ (β .* u .* ui))
    # vi = part .+ (β .* u) #.* ( 1 .- a )
    D[1:72] .= α_a .* ui .* u .* (1 .- u)
    D[73:144] .= -ρ_t * L * v .+ α_t .* Δ .* v .* (1 .- v)
    D[145:216] .= η .* v .* ( 1 .- a )

    # return nothing
end

# --------------------------------------------------------------------------------
# Modelling!
# --------------------------------------------------------------------------------

for (hem_idx, hem) in zip([1:36, 37:72, 1:72], ["right", "left", "all"])
    
    if hem == "right" || hem == "left"
        _cortex = filter(x -> get_hemisphere(x) == hem, cortex)
        atn_model = make_scaled_atn_model((fbb_ui .- fbb_u0)[hem_idx], (part .- v0)[hem_idx], Ld[hem_idx,hem_idx])

    elseif hem == "all"
        _cortex = deepcopy(cortex)
        atn_model = make_scaled_atn_model((fbb_ui .- fbb_u0)[hem_idx], (part .- v0)[hem_idx], Ld[hem_idx,hem_idx])
    end

    inits = [mean_ab_init[hem_idx]; mean_tau_init[hem_idx]; zeros(length(hem_idx))]
    
    ab_tau_coloc_time = calculate_colocalisation_order(_cortex, pst, 3.2258211441306877, atn_model, inits, 0.05, 0.5)

    CSV.write(projectdir("output/analysis-derivatives/colocalisation/0109/colocalisation-inits-order-" * hem * ".csv"), ab_tau_coloc_time)

    ab_tau_coloc_order = calculate_colocalisation_prob(_cortex, pst, 3.2258211441306877, atn_model, inits, 0.05, 0.5)

    CSV.write(projectdir("output/analysis-derivatives/colocalisation/0109/colocalisation-inits-prob-" * hem * ".csv"), ab_tau_coloc_order)
end