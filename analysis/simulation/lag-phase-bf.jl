using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, make_scaled_atn_model_fixed, simulate, make_atn_model,
                                    load_ab_params, load_tau_params, conc, make_scaled_atn_model_hemisphere, 
                                    calculate_colocalisation_order, calculate_colocalisation_prob, find_seed
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names,
                                    get_braak_regions

using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise, sigmoid

using Connectomes: get_hemisphere, plot_roi!, get_node_id, get_lobe, laplacian_matrix, get_label
using Colors, ColorSchemes, GLMakie
using DifferentialEquations
using CSV, DataFrames, DrWatson
using ADNIDatasets
using Statistics
using Serialization
using LinearAlgebra
using DelimitedFiles
using CairoMakie; CairoMakie.activate!()
# --------------------------------------------------------------------------------
# Connectome and Data
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params(tracer="RO")
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c) 
cortex = get_parcellation() |> get_cortex 
subcortex = filter(x -> get_lobe(x) == "subcortex", get_parcellation())[collect(1:10)]
left_subcortex = filter(x -> get_hemisphere(x) == "left", subcortex)
right_subcortex = filter(x -> get_hemisphere(x) == "right", subcortex)

dktnames = get_dkt_names(cortex)
right_cortex = filter(x -> get_hemisphere(x) == "right", cortex)
left_cortex = filter(x -> get_hemisphere(x) == "left", cortex)
# Amyloid data 
include(projectdir("bf-data.jl"))

data_path = datadir("bf-data/bf-data-ab-tau-summary.csv");
data_df = CSV.read(data_path, DataFrame)

ab_data_df = filter(x -> x.ab_status == 1, data_df)
ab_data = BFDataset(ab_data_df, dktnames; min_scans=1, tracer=:ab)

ab_tau_pos_df = filter(x ->  x.ab_status == 1 && x.MTL_Status == 1 && x.NEO_Status == 0, data_df);
tau_data = BFDataset(ab_tau_pos_df, dktnames; min_scans=3, tracer=:tau)
ab_data = BFDataset(ab_tau_pos_df, dktnames; min_scans=3, tracer=:ab)
fmm_u0, fmm_ui = load_ab_params(tracer="FMM")
tau_cutoffs = readdlm(projectdir("output/analysis-derivatives/bf/tau-derivatives/tau-cutoffs-1std-bf.csv")) |> vec

pst = deserialize(projectdir("output/chains/population-atn/pst-samples-bf-random-lognormal-dense-1x1000.jls"));
meanpst = mean(pst)
mean([meanpst["α_a[$i]", :mean] for i in 1:48])
mean([meanpst["ρ_t[$i]", :mean] for i in 1:48])
mean([meanpst["α_t[$i]", :mean] for i in 1:48])
mean([meanpst["η[$i]", :mean] for i in 1:48])
meanpst["β", :mean]
# --------------------------------------------------------------------------------
# Amyloid data
# --------------------------------------------------------------------------------
ab_suvr = calc_suvr.(ab_data)
normalise!(ab_suvr, fmm_u0, fmm_ui)
ab_conc = map(x -> conc.(x, fmm_u0, fmm_ui), ab_suvr)
ab_inits = [d[:,1] for d in ab_conc]

_mean_ab_init = mean(ab_inits)
_mean_ab_init_sym = (_mean_ab_init[1:36] .+ _mean_ab_init[37:end]) ./ 2
mean_ab_init = [_mean_ab_init_sym; _mean_ab_init_sym]
scatter(mean_ab_init)

max_norm(c) =  c ./ maximum(c);

tau_suvr = calc_suvr.(tau_data)
vi = part .+ (4.844 .* (fmm_ui .- fmm_u0))
normalise!(tau_suvr, v0, vi)
tau_conc = map(x -> conc.(x, v0, vi), tau_suvr)
tau_inits = [d[:,1] for d in tau_conc]

_mean_tau_init = mean(tau_inits)
# tau_cutoffs = fill(0.05, 72)
idx = _mean_tau_init .< conc.(tau_cutoffs, v0, vi)
# idx = _mean_tau_init .< tau_cutoffs
_mean_tau_init[idx] .= 0
_mean_tau_init_sym = maximum.(zip(_mean_tau_init[1:36], _mean_tau_init[37:end]))
mean_tau_init = [_mean_tau_init_sym; _mean_tau_init_sym]
scatter(mean_tau_init)

argmax(mean_tau_init)
amyloid_production = 0.18
tau_transport = 0.07
tau_production = 0.07
coupling = 4.844
atrophy = 0.12

p = [amyloid_production, tau_transport, tau_production, coupling, atrophy]

vi = part .+ (coupling .* (fmm_ui .- fmm_u0))
atn_model = make_scaled_atn_model((fmm_ui .- fmm_u0), (part .- v0), L)

prob = ODEProblem(atn_model, [mean_ab_init; mean_tau_init; zeros(72)], (0, 200), p)
sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
d1 = reduce(hcat, [sol(t, Val{1}) for t in 0:0.1:200])
d2 = reduce(hcat, [sol(t, Val{2}) for t in 0:0.1:200])
d3 = reduce(hcat, [sol(t, Val{3}) for t in 0:0.1:200])
d4 = reduce(hcat, [sol(t, Val{4}) for t in 0:0.1:200])

function logistic_solution(t, a, x0)
    return (x0 * exp(a * t))/(1 + x0 * (-1 + exp(a *t)))
end

ab_threshold = Vector{Float64}()
tau_threshold = Vector{Float64}()
tau_threshold_t = Vector{Float64}()
tau_acceleration = Vector{Float64}()
tau_acceleration_t = Vector{Float64}()
for i in 1:72
    ab_dmax = argmax(d1[i, :])
    tau_dmax = argmax(d1[72 + i, :])

    ab_t = argmin(d2[i, :])
    tau_t = argmax(d3[72 + i, 1:tau_dmax])
    push!(tau_threshold_t, tau_t/10)
    tau_t_acc = argmax(d2[72 + i, 1:tau_dmax])
    push!(ab_threshold, sol(ab_t/10)[i])
    push!(tau_threshold, sol(tau_t/10)[72 + i])
    push!(tau_acceleration, sol(tau_t_acc/10)[72 + i])

end
println(mean(ab_threshold))
println(mean(tau_threshold))
scatter(ab_threshold)
scatter(tau_threshold)
writedlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/ab-thresholds-bf.csv"), ab_threshold)
writedlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/tau-thresholds-bf.csv"), tau_threshold)

begin
    f = Figure(size=(1000, 600))
    node = 25
    ax = Axis(f[1,1])
    ylims!(ax, 0, 1)
    plot!(sol, idxs=72+node)
    hlines!(tau_threshold[node])
    hlines!(tau_acceleration[node])
    vlines!(tau_threshold_t[node])
    ax = Axis(f[2,1])
    # ylims!(ax, -0.015, 0.015)
    lines!(0:0.1:80, d2[72+node, :])
    lines!(0:0.1:80, d3[72+node, :])
    # lines!(0:0.1:80, d4[72+node, :])
    vlines!(tau_threshold_t[node])
    f
end

begin
    f = Figure(size=(1000, 600))
    node = 29
    ax = Axis(f[1,1])
    ylims!(ax, 0, 1)
    plot!(sol, idxs=node)
    hlines!(ab_threshold[node])
    # hlines!(ab_acceleration[node])
    # vlines!(ab_threshold_t[node])
    ax = Axis(f[2,1])
    # ylims!(ax, -0.015, 0.015)
    lines!(0:0.1:200, d2[node, :])
    lines!(0:0.1:200, d3[node, :])
    # lines!(0:0.1:80, d4[72+node, :])
    # vlines!(ab_threshold_t[node])
    f
end