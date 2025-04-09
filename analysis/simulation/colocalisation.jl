using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, simulate,
                                    load_ab_params, load_tau_params, conc,
                                    make_atn_prob_func, atn_output_func, make_scaled_atn_model_hemisphere, 
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
_ab_data_df =  CSV.read(datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame) 

tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer && x.AMYLOID_STATUS_COMPOSITE_REF == 1, _ab_data_df);
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF")

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 && x.NEO_Status == 0, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=1)

pst = deserialize(projectdir("output/chains/population-scaled-atn/pst-samples-harmonised-dense-1x1000.jls"));
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
normalise!(tau_suvr, v0, vi)
tau_conc = map(x -> conc.(x, v0, vi), tau_suvr)
tau_inits = [d[:,1] for d in tau_conc]

_mean_tau_init = mean(tau_inits)
_mean_tau_init_sym = (_mean_tau_init[1:36] .+ _mean_tau_init[37:end]) ./ 2
mean_tau_init = [_mean_tau_init_sym; _mean_tau_init_sym]
# mean_tau_init = _mean_tau_init
scatter(mean_tau_init)
filtered_tau_idx = findall(x -> x < 0.05, mean_tau_init)
mean_tau_init[filtered_tau_idx] .= 0

using CairoMakie; CairoMakie.activate!()
scatter(mean_tau_init[1:36])

# --------------------------------------------------------------------------------
# Modelling!
# --------------------------------------------------------------------------------
for (hem_idx, hem) in zip([1:36, 37:72, 1:72], ["right", "left", "all"])
    
    if hem == "right" || hem == "left"
        _cortex = filter(x -> get_hemisphere(x) == hem, cortex)
        atn_model = make_scaled_atn_model_hemisphere((fbb_ui .- fbb_u0)[hem_idx], (part .- v0)[hem_idx], L[hem_idx,hem_idx])

    elseif hem == "all"
        _cortex = deepcopy(cortex)
        atn_model = make_scaled_atn_model((fbb_ui .- fbb_u0)[hem_idx], (part .- v0)[hem_idx], L[hem_idx,hem_idx])
    end

    inits = [mean_ab_init[hem_idx]; mean_tau_init[hem_idx]; zeros(length(hem_idx))]
    
    ab_tau_coloc_time = calculate_colocalisation_order(_cortex, pst, atn_model, inits, 0.1, 0.75)

    CSV.write(projectdir("output/analysis-derivatives/colocalisation/01075/colocalisation-inits-order-" * hem * ".csv"), ab_tau_coloc_time)

    ab_tau_coloc_order = calculate_colocalisation_prob(_cortex, pst, atn_model, inits, 0.1, 0.75)

    CSV.write(projectdir("output/analysis-derivatives/colocalisation/01075/colocalisation-inits-prob-" * hem * ".csv"), ab_tau_coloc_order)
end