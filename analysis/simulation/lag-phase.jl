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
using Turing
using CairoMakie; CairoMakie.activate!()
# --------------------------------------------------------------------------------
# Connectome and Data
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params()
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
_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
# tau_pos_df = filter(x ->  x.MTL_Status == 1 && x.NEO_Status == 0, tau_data_df)
# tau_pos_df = filter(x ->  x.MTL_Status == 1 , tau_data_df)
_tau_data = ADNIDataset(tau_data_df, dktnames; min_scans=1)
tau_subs = get_id.(_tau_data)
tau_cutoffs = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-1std.csv")) |> vec

tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer 
                && x.AMYLOID_STATUS_COMPOSITE_REF == 1 && x.RID 
                ∈ tau_subs && x.CENTILOIDS < 70, _ab_data_df);

amy_pos_init_idx = [findfirst(isequal(id), _ab_data_df.RID) for id in unique(fbb_data_df.RID)]
pos_centiloids = mean(_ab_data_df[amy_pos_init_idx, :].CENTILOIDS)
pos_centiloids_st = std(_ab_data_df[amy_pos_init_idx, :].CENTILOIDS)

fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF")

tau_data = filter(x -> get_id(x) ∈ get_id.(fbb_data), _tau_data)

pst = chainscat([deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-random-beta-lognormal-4x1000-$i.jls")) for i in 1:4]...)

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
scatter(mean_ab_init)

max_norm(c) =  c ./ maximum(c);

tau_suvr = calc_suvr.(tau_data)
vi = part .+ (4.836840700255846.* (fbb_ui .- fbb_u0))
normalise!(tau_suvr, v0, vi)
tau_conc = map(x -> conc.(x, v0, vi), tau_suvr)
tau_inits = [d[:,1] for d in tau_conc]

_mean_tau_init = mean(tau_inits)
idx = _mean_tau_init .< conc.(tau_cutoffs, v0, vi)
_mean_tau_init[idx] .= 0
_mean_tau_init_sym = mean.(zip(_mean_tau_init[1:36], _mean_tau_init[37:end]))
mean_tau_init = [_mean_tau_init_sym; _mean_tau_init_sym]
scatter(mean_tau_init)

amyloid_production = mean([meanpst["α_a[$i]", :mean] for i in 1:18])
tau_transport = mean([meanpst["ρ_t[$i]", :mean] for i in 1:18])
tau_production = mean([meanpst["α_t[$i]", :mean] for i in 1:18])
coupling = meanpst["β_fbb", :mean]
atrophy = mean([meanpst["η[$i]", :mean] for i in 1:18])

p = [amyloid_production, tau_transport, tau_production, coupling, atrophy]

vi = part .+ (coupling .* (fbb_ui .- fbb_u0))
atn_model = make_scaled_atn_model((fbb_ui .- fbb_u0), (part .- v0), L)

prob = ODEProblem(atn_model, [mean_ab_init; mean_tau_init; zeros(72)], (0, 80), p)
sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
d1 = reduce(hcat, [sol(t, Val{1}) for t in 0:0.1:80])
d2 = reduce(hcat, [sol(t, Val{2}) for t in 0:0.1:80])
d3 = reduce(hcat, [sol(t, Val{3}) for t in 0:0.1:80])
d4 = reduce(hcat, [sol(t, Val{4}) for t in 0:0.1:80])

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
writedlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/ab-thresholds.csv"), ab_threshold)
writedlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/tau-thresholds.csv"), tau_threshold)