using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, simulate, make_atn_model,
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
tau_pos_df = filter(x ->  x.MTL_Status == 1 && x.NEO_Status == 0, tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1, tau_data_df);
_tau_data = ADNIDataset(tau_data_df, dktnames; min_scans=1)
tau_subs = get_id.(_tau_data)

tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer && x.AMYLOID_STATUS_COMPOSITE_REF == 1 
                            && x.RID ∈ tau_subs && x.CENTILOIDS < 65, _ab_data_df);
                            # && x.RID ∈ tau_subs && x.CENTILOIDS < 60, _ab_data_df);
# fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer && x.AMYLOID_STATUS_COMPOSITE_REF == 1 && x.RID ∈ tau_subs, _ab_data_df);
mean(fbb_data_df.CENTILOIDS)
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF")

tau_data = filter(x -> get_id(x) ∈ get_id.(fbb_data), _tau_data)

tau_cutoffs = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-1std.csv")) |> vec
# pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-sepbeta-1x1000.jls"));
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
# mean_ab_init = _mean_ab_init
scatter(mean_ab_init)
max_norm(c) =  c ./ maximum(c);
# --------------------------------------------------------------------------------
# Tau data
# --------------------------------------------------------------------------------
tau_suvr = calc_suvr.(tau_data)
vi = part .+ (meanpst["β_fbb",:mean].* (fbb_ui .- fbb_u0))
normalise!(tau_suvr, v0, vi)
tau_conc = map(x -> conc.(x, v0, vi), tau_suvr)
tau_inits = [d[:,1] for d in tau_conc]

_mean_tau_init = mean(tau_inits)
# tau_cutoffs= fill(0.05, 72)
idx = _mean_tau_init .< conc.(tau_cutoffs, v0, vi)
# idx = _mean_tau_init .< tau_cutoffs
_mean_tau_init[idx] .= 0
_mean_tau_init_sym = mean.(zip(_mean_tau_init[1:36], _mean_tau_init[37:end]))
mean_tau_init = [_mean_tau_init_sym; _mean_tau_init_sym]
non_seed_idx = findall(x -> x == 0, mean_tau_init)
using CairoMakie; CairoMakie.activate!()
scatter(mean_tau_init)

ab_threshold = readdlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/ab-thresholds.csv")) |> vec
tau_threshold = readdlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/tau-thresholds.csv")) |> vec
# ab_threshold = fill(mean(ab_threshold), 72)
# tau_threshold = fill(mean(tau_threshold[non_seed_idx]), 72)
# ab_threshold = fill(0.5,72)
# --------------------------------------------------------------------------------
# Modelling!
# --------------------------------------------------------------------------------
for (hem_idx, hem) in zip([1:36, 37:72, 1:72], ["right", "left", "all"])
# for (hem_idx, hem) in zip([1:72], ["all"])
    
    if hem == "right" || hem == "left"
        _cortex = filter(x -> get_hemisphere(x) == hem, cortex)
        _ab_threshold = ab_threshold[hem_idx]
        _tau_threshold = tau_threshold[hem_idx]
        atn_model = make_scaled_atn_model((fbb_ui .- fbb_u0)[hem_idx], (part .- v0)[hem_idx], L[hem_idx,hem_idx])
    elseif hem == "all"
        _cortex = deepcopy(cortex)
        _ab_threshold = ab_threshold[hem_idx]
        _tau_threshold = tau_threshold[hem_idx]
        atn_model = make_scaled_atn_model((fbb_ui .- fbb_u0)[hem_idx], (part .- v0)[hem_idx], L[hem_idx,hem_idx])
    end

    inits = [mean_ab_init[hem_idx]; mean_tau_init[hem_idx]; zeros(length(hem_idx))]
    
    # ab_tau_coloc_time = calculate_colocalisation_order(_cortex, pst, 3.2258211441306877, atn_model, inits, _tau_threshold, _ab_threshold)
    # display(ab_tau_coloc_time)
    # CSV.write(projectdir("output/analysis-derivatives/colocalisation/0175/colocalisation-inits-order-threshold" * hem * ".csv"), ab_tau_coloc_time)
    # ab_tau_coloc_time = calculate_colocalisation_order(_cortex, pst, atn_model, inits, 0.09, 0.79)
    ab_tau_coloc_time = calculate_colocalisation_order(_cortex, pst, atn_model, inits, _tau_threshold, _ab_threshold; tracer=:β_fbb)
    display(ab_tau_coloc_time)
    CSV.write(projectdir("output/analysis-derivatives/colocalisation/ab-tau-thresholds-random/colocalisation-inits-order-" * hem * ".csv"), ab_tau_coloc_time)

    # ab_tau_coloc_order = calculate_colocalisation_prob(_cortex, pst, 3.2258211441306877, atn_model, inits, _tau_threshold, _ab_threshold)
    # display(ab_tau_coloc_order)
    # CSV.write(projectdir("output/analysis-derivatives/colocalisation/0175/colocalisation-inits-prob-threshold" * hem * ".csv"), ab_tau_coloc_order)
    # ab_tau_coloc_order = calculate_colocalisation_prob(_cortex, pst, atn_model, inits, 0.09, 0.79)
    ab_tau_coloc_order = calculate_colocalisation_prob(_cortex, pst, atn_model, inits, _tau_threshold, _ab_threshold; tracer=:β_fbb)
    display(ab_tau_coloc_order)
    CSV.write(projectdir("output/analysis-derivatives/colocalisation/ab-tau-thresholds-random/colocalisation-inits-prob-" * hem * ".csv"), ab_tau_coloc_order)
end

tau_threshold = readdlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/tau-thresholds.csv")) |> vec
ab_threshold = fill(0.5, 72)
# ab_threshold = fill(mean(ab_threshold), 72)
# tau_threshold = fill(mean(tau_threshold[non_seed_idx]), 72)
# ab_threshold = fill(0.5,72)
# --------------------------------------------------------------------------------
# Modelling!
# --------------------------------------------------------------------------------
for (hem_idx, hem) in zip([1:36, 37:72, 1:72], ["right", "left", "all"])
# for (hem_idx, hem) in zip([1:72], ["all"])
    
    if hem == "right" || hem == "left"
        _cortex = filter(x -> get_hemisphere(x) == hem, cortex)
        _ab_threshold = ab_threshold[hem_idx]
        _tau_threshold = tau_threshold[hem_idx]
        atn_model = make_scaled_atn_model((fbb_ui .- fbb_u0)[hem_idx], (part .- v0)[hem_idx], L[hem_idx,hem_idx])
    elseif hem == "all"
        _cortex = deepcopy(cortex)
        _ab_threshold = ab_threshold[hem_idx]
        _tau_threshold = tau_threshold[hem_idx]
        atn_model = make_scaled_atn_model((fbb_ui .- fbb_u0)[hem_idx], (part .- v0)[hem_idx], L[hem_idx,hem_idx])
    end

    inits = [mean_ab_init[hem_idx]; mean_tau_init[hem_idx]; zeros(length(hem_idx))]
    
    # ab_tau_coloc_time = calculate_colocalisation_order(_cortex, pst, 3.2258211441306877, atn_model, inits, _tau_threshold, _ab_threshold)
    # display(ab_tau_coloc_time)
    # CSV.write(projectdir("output/analysis-derivatives/colocalisation/0175/colocalisation-inits-order-threshold" * hem * ".csv"), ab_tau_coloc_time)
    # ab_tau_coloc_time = calculate_colocalisation_order(_cortex, pst, atn_model, inits, 0.09, 0.79)
    ab_tau_coloc_time = calculate_colocalisation_order(_cortex, pst, atn_model, inits, _tau_threshold, _ab_threshold; tracer=:β_fbb)
    display(ab_tau_coloc_time)
    CSV.write(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-order-" * hem * ".csv"), ab_tau_coloc_time)

    # ab_tau_coloc_order = calculate_colocalisation_prob(_cortex, pst, 3.2258211441306877, atn_model, inits, _tau_threshold, _ab_threshold)
    # display(ab_tau_coloc_order)
    # CSV.write(projectdir("output/analysis-derivatives/colocalisation/0175/colocalisation-inits-prob-threshold" * hem * ".csv"), ab_tau_coloc_order)
    # ab_tau_coloc_order = calculate_colocalisation_prob(_cortex, pst, atn_model, inits, 0.09, 0.79)
    ab_tau_coloc_order = calculate_colocalisation_prob(_cortex, pst, atn_model, inits, _tau_threshold, _ab_threshold; tracer=:β_fbb)
    display(ab_tau_coloc_order)
    CSV.write(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-prob-" * hem * ".csv"), ab_tau_coloc_order)
end
# using CSV, DataFrames, CairoMakie
# using DrWatson
# df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-order-right.csv"), DataFrame)
# df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-prob-right.csv"), DataFrame)
# df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-order-left.csv"), DataFrame)
# df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-prob-left.csv"), DataFrame)

# scatter(df.Coloc_time, df.tau_time)

# using GLMakie, Connectomes, ColorSchemes

# d = (df.tau_time .- minimum(df.tau_time)) ./ (maximum(df.tau_time) .- minimum(df.tau_time) )

# plot_roi(df.DKTID, d, reverse(ColorSchemes.viridis))