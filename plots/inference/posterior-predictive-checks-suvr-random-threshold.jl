using ATNModelling.SimulationUtils: make_prob, make_atn_model,
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params, conc
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names,
                                    get_braak_regions
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise, make_ucsf_df,rename_ucsf_df, 
                              add_icv, make_ucsf_name, make_dkt_name, normalise_and_threshold!
using ATNModelling.InferenceModels: fit_model, ensemble_atn, serial_atn, fit_serial_catn

using Connectomes: laplacian_matrix, get_label, get_node_id
using ADNIDatasets: ADNIDataset, get_id, get_dates, get_initial_conditions, 
                    calc_suvr, get_vol, get_times, data_dashboard
using DrWatson: projectdir, datadir
using CSV, DataFrames
using SciMLBase: successful_retcode
using DifferentialEquations, Turing, LinearAlgebra
using Random
using StatisticalMeasures
using Serialization
using LsqFit
using CairoMakie; CairoMakie.activate!()
using Colors, ColorSchemes
using DelimitedFiles
include(projectdir("bf-data.jl"))
# --------------------------------------------------------------------------------
# Tracer independent data
# --------------------------------------------------------------------------------
ftp_v0, ftp_vi, ftp_part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c) 
dktnames = get_parcellation() |> get_cortex |> get_dkt_names

# Amyloid data 
_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=3)

ucsf_df = CSV.read(datadir("ADNI/2025/adni-ucsf-crosssec-mri.csv"), DataFrame)
data_dict_df = CSV.read(datadir("ADNI/adni-data-dictionary.csv"), DataFrame)
ucsf_data_dict = filter(x -> x.CRFNAME == "Cross-Sectional FreeSurfer (7.x)", data_dict_df)
atr_df = make_ucsf_df(ucsf_df, ucsf_data_dict, dktnames)
filter!(x -> x.OVERALLQC != "Fail", atr_df)
tau_atr_df = add_icv(tau_pos_df, atr_df; dt_threshold=180)
tau_icv_df = filter(x -> x.Has_ICV, tau_atr_df)

tau_data = ADNIDataset(tau_icv_df, dktnames; min_scans=3)

tau_thresholds = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-1std.csv")) |> vec
# --------------------------------------------------------------------------------
# Load fbb data
# --------------------------------------------------------------------------------
tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

fbb, fbb_tau = align_data(fbb_data, tau_data; min_tau_scans=3)

fbb_times = get_times.(fbb)
fbb_tau_times = get_times.(fbb_tau)
fbb_ts = [sort(unique([a; t])) for (a, t) in zip(fbb_times, fbb_tau_times)]

fbb_ab_tidx = get_time_idx(fbb_times, fbb_ts)
fbb_tau_tidx = get_time_idx(fbb_tau_times, fbb_ts)

@assert get_id.(fbb) == get_id.(fbb_tau)
@assert allequal([allequal(fbb_times[i] .== fbb_ts[i][fbb_ab_tidx[i]]) for i in 1:length(fbb)])
@assert allequal([allequal(fbb_tau_times[i] .== fbb_ts[i][fbb_tau_tidx[i]]) for i in 1:length(fbb_tau)])

fbb_suvr = calc_suvr.(fbb)
normalise!(fbb_suvr, fbb_u0, fbb_ui)
fbb_inits = [d[:,1] for d in fbb_suvr]

fbb_tau_suvr = calc_suvr.(fbb_tau)
normalise_and_threshold!(fbb_tau_suvr, ftp_v0, tau_thresholds)
fbb_tau_inits = [d[:,1] for d in fbb_tau_suvr]

fbb_tau_pos_vol = get_vol.(fbb_tau)
fbb_tau_pos_icv = [filter(x -> x.RID == rid, tau_icv_df).ICV for rid in get_id.(fbb_tau)]
allequal(length.(fbb_tau_pos_icv) .== size.(calc_suvr.(fbb_tau), 2))
fbb_total_vol_norm = [ v ./ t' for (v, t) in zip(fbb_tau_pos_vol, fbb_tau_pos_icv)]
fbb_vols = [clamp.(1 .- (vol ./ vol[:,1]), 0, 1) for vol in fbb_total_vol_norm]
fbb_vol_inits = [vol[:,1] for vol in fbb_vols]

fbb_inits = [[ab; tau; vol] for (ab, tau, vol) in zip(fbb_inits, fbb_tau_inits, fbb_vol_inits)]
fbb_n = length(fbb)

fbb_atn_model = make_atn_model(fbb_u0, fbb_ui, ftp_v0, ftp_part, L)
fbb_prob = make_prob(fbb_atn_model, fbb_inits[1], (0.0,7.5), [1.0,0.1,1.0,3.5,1.0])
sol = solve(fbb_prob, Tsit5())
# --------------------------------------------------------------------------------
# Load fbp data
# --------------------------------------------------------------------------------
tracer="FBP"
fbp_u0, fbp_ui = load_ab_params(tracer=tracer)
fbp_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
fbp_data = ADNIDataset(fbp_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

fbp, fbp_tau = align_data(fbp_data, tau_data; min_tau_scans=3)

@assert allequal([x ∉ get_id.(fbp_tau) for x in get_id.(fbb_tau)])

fbp_times = get_times.(fbp)
fbp_tau_times = get_times.(fbp_tau)
fbp_ts = [sort(unique([a; t])) for (a, t) in zip(fbp_times, fbp_tau_times)]

fbp_ab_tidx = get_time_idx(fbp_times, fbp_ts)
fbp_tau_tidx = get_time_idx(fbp_tau_times, fbp_ts)

@assert get_id.(fbp) == get_id.(fbp_tau)
@assert allequal([allequal(fbp_times[i] .== fbp_ts[i][fbp_ab_tidx[i]]) for i in 1:length(fbp)])
@assert allequal([allequal(fbp_tau_times[i] .== fbp_ts[i][fbp_tau_tidx[i]]) for i in 1:length(fbp_tau)])

fbp_suvr = calc_suvr.(fbp)
normalise!(fbp_suvr, fbp_u0, fbp_ui)
fbp_inits = [d[:,1] for d in fbp_suvr]

fbp_tau_suvr = calc_suvr.(fbp_tau)
normalise_and_threshold!(fbp_tau_suvr, ftp_v0, tau_thresholds)
fbp_tau_inits = [d[:,1] for d in fbp_tau_suvr]

fbp_tau_pos_vol = get_vol.(fbp_tau)
fbp_tau_pos_icv = [filter(x -> x.RID == rid, tau_icv_df).ICV for rid in get_id.(fbp_tau)]
allequal(length.(fbp_tau_pos_icv) .== size.(calc_suvr.(fbp_tau), 2))
fbp_total_vol_norm = [ v ./ t' for (v, t) in zip(fbp_tau_pos_vol, fbp_tau_pos_icv)]
fbp_vols = [clamp.(1 .- (vol ./ vol[:,1]), 0, 1) for vol in fbp_total_vol_norm]
fbp_vol_inits = [vol[:,1] for vol in fbp_vols]

fbp_inits = [[ab; tau; vol] for (ab, tau, vol) in zip(fbp_inits, fbp_tau_inits, fbp_vol_inits)]
fbp_n = length(fbp)

fbp_atn_model = make_atn_model(fbp_u0, fbp_ui, ftp_v0, ftp_part, L)
fbp_prob = make_prob(fbp_atn_model, fbp_inits[1], (0.0,7.5), [1.0,0.1,1.0,3.5,1.0])
sol = solve(fbp_prob, Tsit5())
# --------------------------------------------------------------------------------
# Load BF data
# --------------------------------------------------------------------------------
tracer="FMM"
bf_u0, bf_ui = load_ab_params(tracer=tracer)
bf_v0, bf_vi, bf_part = load_tau_params(tracer="RO")

bf_data_path = datadir("bf-data/bf-data-ab-tau-summary.csv");
bf_data_df = CSV.read(bf_data_path, DataFrame)

bf_ab_data_df = filter(x -> x.ab_status == 1, bf_data_df)
bf_ab_data = BFDataset(bf_ab_data_df, dktnames; min_scans=3, tracer=:ab)

# Tau data 
bf_tau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, bf_ab_data_df);
bf_tau_data = BFDataset(bf_tau_pos_df, dktnames; min_scans=3, tracer=:tau)

bf_ab, bf_tau = align_data(bf_ab_data, bf_tau_data)

bf_ab_times = get_times.(bf_ab)
bf_tau_times = get_times.(bf_tau)
bf_ts = [sort(unique([a; t])) for (a, t) in zip(bf_ab_times, bf_tau_times)]

bf_ab_tidx = get_time_idx(bf_ab_times, bf_ts)
bf_tau_tidx = get_time_idx(bf_tau_times, bf_ts)

@assert allequal([allequal(bf_ab_times[i] .== bf_ts[i][bf_ab_tidx[i]]) for i in 1:48])
@assert allequal([allequal(bf_tau_times[i] .== bf_ts[i][bf_tau_tidx[i]]) for i in 1:48])

bf_ab_suvr = calc_suvr.(bf_ab)
normalise!(bf_ab_suvr, bf_u0, bf_ui)
bf_ab_inits = [d[:,1] for d in bf_ab_suvr]

bf_tau_suvr = calc_suvr.(bf_tau)
normalise!(bf_tau_suvr, bf_v0)
bf_tau_inits = [d[:,1] for d in bf_tau_suvr]

bf_tau_pos_vol = get_vol.(bf_tau)
bf_tau_pos_icv = [filter(x -> x.sid == "BF" * string(get_id(tau)) && x.tau_pet_date ∈ get_dates(tau), bf_data_df).icv_mm3 for tau in bf_tau]
dates = [filter(x -> x.sid == "BF" * string(get_id(tau)) && x.tau_pet_date ∈ get_dates(tau), bf_data_df).tau_pet_date for tau in bf_tau]
@assert allequal(reduce(vcat, [d .== d1 for (d, d1) in zip(dates, get_dates.(bf_tau))]))
@assert allequal(length.(bf_tau_pos_icv) .== size.(calc_suvr.(bf_tau), 2))
bf_total_vol_norm = [ v ./ t' for (v, t) in zip(bf_tau_pos_vol, bf_tau_pos_icv)]
bf_vols = [clamp.(1 .- (vol ./ vol[:,1]), 0, 1) for vol in bf_total_vol_norm]
bf_vols[1]
bf_vol_inits = [vol[:,1] for vol in bf_vols]

bf_inits = [[ab; tau; vol] for (ab, tau, vol) in zip(bf_ab_inits, bf_tau_inits, bf_vol_inits)]
bf_n = length(bf_ab)

bf_atn_model = make_atn_model(bf_u0, bf_ui, bf_v0, bf_part, L)
bf_prob = make_prob(bf_atn_model, bf_inits[1], (0.0,7.5), [1.0,0.1,1.0,3.5,1.0])
sol = solve(bf_prob, Tsit5())
plot(sol, idxs=73:144)
# --------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------
# pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-fixed-beta-1x1000.jls"));
# adni_pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-fixed-beta-lognormal-1x1000.jls"));
# bf_pst = deserialize(projectdir("output/chains/population-atn/pst-samples-bf-fixed-beta-lognormal-1x1000.jls"));
adni_pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-random-beta-lognormal-threshold-1x1000-1.jls"));
# adni_pst = chainscat([deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-random-beta-lognormal-4x1000-$i.jls")) for i in 1:4]...)
bf_pst = deserialize(projectdir("output/chains/population-atn/pst-samples-bf-random-lognormal-dense-1x1000.jls"));

summarize(adni_pst)
summarize(bf_pst)
adni_meanpst = mean(adni_pst)
bf_meanpst = mean(bf_pst)

# bf_results = CSV.read(projectdir("output/bf-output/bf/averaged_results.csv"), DataFrame)

function get_diff(d)
    d[:,end] .- d[:,1]
end

linearmodel(x, p) = bf_part .+ p[1] .* x
fbp_fitted_model = curve_fit(linearmodel, fbp_ui .- fbp_u0, ftp_vi, [1.0])
# println("params = $(fbp_fitted_model.param)")


bf_fitted_model = curve_fit(linearmodel, bf_ui .- bf_u0, bf_vi, [1.0])
println("params = $(bf_fitted_model.param)")

# using GLMakie, Colors, ColorSchemes, Connectomes
# data_dashboard(tau, v0, vi; show_mtl_threshold=true)

using CairoMakie; CairoMakie.activate!()

braak_regions = get_braak_regions()
bs = [findall(x -> get_node_id(x) ∈ br, cortex) for br in braak_regions]

ab_sols = Vector{Array{Float64}}()
tau_sols = Vector{Array{Float64}}()
atr_sols = Vector{Array{Float64}}()
adni_meanpst = mean(adni_pst)

for sub in 1:18
#     p = [adni_meanpst[Symbol("α_a[$sub]"), :mean], adni_meanpst[Symbol("ρ_t[$sub]"), :mean], 
#     adni_meanpst[Symbol("α_t[$sub]"), :mean], 3.2258211441306877, 
#     adni_meanpst[Symbol("η[$sub]"), :mean]]
    p = [adni_meanpst[Symbol("α_a[$sub]"), :mean], adni_meanpst[Symbol("ρ_t[$sub]"), :mean], 
    adni_meanpst[Symbol("α_t[$sub]"), :mean], adni_meanpst["β_fbb", :mean], 
    adni_meanpst[Symbol("η[$sub]"), :mean]]
    
    pstprob = remake(fbb_prob, u0=fbb_inits[sub], p=p)
    pstsol = solve(pstprob, Tsit5())
    push!(ab_sols,pstsol(fbb_times[sub])[1:72,:])
    push!(tau_sols,pstsol(fbb_tau_times[sub])[73:144,:])
    push!(atr_sols,pstsol(fbb_tau_times[sub])[145:216,:])
end

for (i, sub) in enumerate(19:34)
#     p = [adni_meanpst[Symbol("α_a[$sub]"), :mean], adni_meanpst[Symbol("ρ_t[$sub]"), :mean], 
#     adni_meanpst[Symbol("α_t[$sub]"), :mean], 3.685830432811308, 
#     adni_meanpst[Symbol("η[$sub]"), :mean]]

        p = [adni_meanpst[Symbol("α_a[$sub]"), :mean], adni_meanpst[Symbol("ρ_t[$sub]"), :mean], 
    adni_meanpst[Symbol("α_t[$sub]"), :mean], adni_meanpst["β_fbp", :mean], 
    adni_meanpst[Symbol("η[$sub]"), :mean]]
    
    pstprob = remake(fbp_prob, u0=fbp_inits[i], p=p)
    pstsol = solve(pstprob, Tsit5())
    push!(ab_sols,  pstsol(fbp_times[i])[1:72,:])
    push!(tau_sols, pstsol(fbp_tau_times[i])[73:144,:])
    push!(atr_sols, pstsol(fbp_tau_times[i])[145:216,:])
end

mean_ab_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in ab_sols]), dims=2))
mean_tau_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in tau_sols]), dims=2))
mean_atr_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in atr_sols]), dims=2))

mean_ab =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_suvr; fbp_suvr]]), dims=2))
mean_tau =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_tau_suvr; fbp_tau_suvr]]), dims=2))
mean_atr =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_vols; fbp_vols]]), dims=2))

mean_ab_sol_diff = vec(mean(reduce(hcat, get_diff.(ab_sols)), dims=2))
mean_tau_sol_diff = vec(mean(reduce(hcat, get_diff.(tau_sols)), dims=2))
mean_atr_sol_diff = vec(mean(reduce(hcat, get_diff.(atr_sols)), dims=2))

mean_ab_diff = vec(mean(reduce(hcat, get_diff.([fbb_suvr; fbp_suvr])), dims=2))
mean_tau_diff = vec(mean(reduce(hcat, get_diff.([fbb_tau_suvr; fbp_tau_suvr])), dims=2))
mean_atr_diff = vec(mean(reduce(hcat, get_diff.([fbb_vols; fbp_vols])), dims=2))

adni_results = DataFrame(
        mean_ab_sols = mean_ab_sols, 
        mean_tau_sols = mean_tau_sols,
        mean_atr_sols = mean_atr_sols,
        mean_ab = mean_ab,
        mean_tau = mean_tau,
        mean_atr = mean_atr,
        mean_ab_sol_diff = mean_ab_sol_diff,
        mean_tau_sol_diff = mean_tau_sol_diff,
        mean_atr_sol_diff = mean_atr_sol_diff,
        mean_ab_diff = mean_ab_diff,
        mean_tau_diff = mean_tau_diff,
        mean_atr_diff = mean_atr_diff
)


bf_ab_sols = Vector{Array{Float64}}()
bf_tau_sols = Vector{Array{Float64}}()
bf_atr_sols = Vector{Array{Float64}}()
bf_meanpst = mean(bf_pst)

for sub in 1:48
#     p = [bf_meanpst[Symbol("α_a[$sub]"), :mean], bf_meanpst[Symbol("ρ_t[$sub]"), :mean], 
#     bf_meanpst[Symbol("α_t[$sub]"), :mean], 2.2182243777142356, 
#     bf_meanpst[Symbol("η[$sub]"), :mean]]
    
    p = [bf_meanpst[Symbol("α_a[$sub]"), :mean], bf_meanpst[Symbol("ρ_t[$sub]"), :mean], 
    bf_meanpst[Symbol("α_t[$sub]"), :mean], bf_meanpst[Symbol("β"), :mean],
    bf_meanpst[Symbol("η[$sub]"), :mean]]
    
    pstprob = remake(bf_prob, u0=bf_inits[sub], p=p)
    pstsol = solve(pstprob, Tsit5(), abstol=1e-12, reltol=1e-12)
    push!(bf_ab_sols,pstsol(bf_ab_times[sub])[1:72,:])
    push!(bf_tau_sols,pstsol(bf_tau_times[sub])[73:144,:])
    push!(bf_atr_sols,pstsol(bf_tau_times[sub])[145:216,:])
end

bf_mean_ab_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in bf_ab_sols]), dims=2))
bf_mean_tau_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in bf_tau_sols]), dims=2))
bf_mean_atr_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in bf_atr_sols]), dims=2))

bf_mean_ab =  vec(mean(reduce(hcat, [d[:, end] for d in bf_ab_suvr]), dims=2))
bf_mean_tau =  vec(mean(reduce(hcat, [d[:, end] for d in bf_tau_suvr]), dims=2))
bf_mean_atr =  vec(mean(reduce(hcat, [d[:, end] for d in bf_vols]), dims=2))

bf_mean_ab_sol_diff = vec(mean(reduce(hcat, get_diff.(bf_ab_sols)), dims=2))
bf_mean_tau_sol_diff = vec(mean(reduce(hcat, get_diff.(bf_tau_sols)), dims=2))
bf_mean_atr_sol_diff = vec(mean(reduce(hcat, get_diff.(bf_atr_sols)), dims=2))

bf_mean_ab_diff = vec(mean(reduce(hcat, get_diff.(bf_ab_suvr)), dims=2))
bf_mean_tau_diff = vec(mean(reduce(hcat, get_diff.(bf_tau_suvr)), dims=2))
bf_mean_atr_diff = vec(mean(reduce(hcat, get_diff.(bf_vols)), dims=2))


bf_results = DataFrame(
        mean_ab_sols = bf_mean_ab_sols, 
        mean_tau_sols = bf_mean_tau_sols,
        mean_atr_sols = bf_mean_atr_sols,
        mean_ab = bf_mean_ab,
        mean_tau = bf_mean_tau,
        mean_atr = bf_mean_atr,
        mean_ab_sol_diff = bf_mean_ab_sol_diff,
        mean_tau_sol_diff = bf_mean_tau_sol_diff,
        mean_atr_sol_diff = bf_mean_atr_sol_diff,
        mean_ab_diff = bf_mean_ab_diff,
        mean_tau_diff = bf_mean_tau_diff,
        mean_atr_diff = bf_mean_atr_diff
)
using GLM 
# CSV.write(projectdir("output/analysis-derivatives/posterior-derivatives/averaged_results_adni.csv"), averaged_results_adni)
# adni_results = CSV.read(projectdir("output/analysis-derivatives/posterior-derivatives/averaged_results_adni.csv"), DataFrame)
# bf_results = CSV.read(projectdir("output/bf-output/bf/averaged_results.csv"), DataFrame)

# Label(f[0, 1], "Aβ", tellwidth=false, fontsize=35)
# Label(f[0, 2], "Tau", tellwidth=false, fontsize=35)
# Label(f[0, 3], "Neurodegeneration", tellwidth=false, fontsize=35)
# Label(f[1, 0], "ADNI", tellwidth=true, tellheight=false, fontsize=35)
# Label(f[2, 0], "BF-2", tellwidth=true, tellheight=false, fontsize=35)

# rois = reduce(vcat, bs[1:3])

_rois = ["entorhinal", "Left-Hippocampus", "Right-Hippocampus", "Left-Amygdala", "Right-Amygdala",
                "inferiortemporal", "middletemporal", "inferiorparietal", "precuneus"]
rois = findall(x -> x ∈ _rois, get_label.(cortex))

# pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-fixed-beta-1x1000.jls"));
# bf_pst = deserialize(projectdir("output/chains/population-atn/pst-samples-suvr-bf-fixed-beta-1x1000.jls"));
braak_regions = get_braak_regions()
bs = [findall(x -> get_node_id(x) ∈ br, cortex) for br in braak_regions]

begin
        cmap = Makie.wong_colors();
        ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
        tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
        atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);
        abcmap = ColorScheme(ab_c);
        taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
        atrcmap = ColorScheme(atr_c); #ColorSchemes.Reds;
    
        f = Figure(size=(1500, 1200))

        titlesize = 30
        xlabelsize = 30
        ylabelsize = 25
        xticklabelsize = 25 
        yticklabelsize = 25
        rsize=30

        g0 = f[1:4,1] = GridLayout()
        g1 = f[1:2,2] = GridLayout()
        
        ax = Axis(g1[1,1], xticks=0:0.2:0.4,   xticklabelsize=25, xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, title="α \n Aβ production", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20, titlefont=:regular)
        hidespines!(ax, :l, :t, :r)
        hideydecorations!(ax, label=false)
        CairoMakie.xlims!(ax, 0.0,0.5)
        hist!(vec(Array(adni_pst[:Am_a])), bins=15, color=alphacolor(get(abcmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
        hidexdecorations!(ax, grid=false, ticks=false)
    
        ax = Axis(g1[1,2], xticks=0:0.04:0.08,  xticklabelsize=25, xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, title="ρ \n tau transport", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20, titlefont=:regular)
        hidespines!(ax, :l, :t, :r)
        hideydecorations!(ax, label=false)
        xlims!(ax, 0.0,0.1)
        hist!(vec(Array(adni_pst[:Pm_t])), bins=15, color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
        hidexdecorations!(ax, grid=false, ticks=false)
    
        ax = Axis(g1[1,3],  xticks=0:0.05:0.1,  xticklabelsize=25, xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, title="γ \n tau production", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20, titlefont=:regular)
        hidespines!(ax, :l, :t, :r)
        hideydecorations!(ax, label=false)
        xlims!(ax, 0.0,0.125)
        hist!(vec(Array(adni_pst[:Am_t])), bins=15,color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
        hidexdecorations!(ax, grid=false, ticks=false)
    
    
        ax = Axis(g1[1,4], xticks=4:0.5:6, xticklabelsize=25, titlesize=titlesize, title="β \n Aβ/tau coupling", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20, titlefont=:regular)
        hidespines!(ax, :l, :t, :r)
        hideydecorations!(ax, label=false)
        xlims!(ax, 3.5, 6.5)
        hist!(vec(Array(adni_pst[:β_fbb])), bins=10,color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1, label="FBB")
        hist!(vec(Array(adni_pst[:β_fbp])), bins=10,color=alphacolor(get(taucmap, 0.5), 1.0), strokecolor=:white, strokewidth=1, label="FBP")
        hidexdecorations!(ax, grid=false, ticks=false)
        axislegend(ax, position=:lt, labelsize=20)

        ax = Axis(g1[1,5], xticks=0.0:0.05:0.2,  xticklabelsize=25, xlabel=L"1 / yr", xlabelsize=25,titlesize=titlesize, title="η \n atrophy rate", ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20, titlefont=:regular)
        hidespines!(ax, :l, :t, :r)
        hideydecorations!(ax, label=false)
        hidexdecorations!(ax, grid=false, ticks=false)
        xlims!(ax, 0.0,0.22)
        hist!(vec(Array(adni_pst[:Em])), bins=15, color=alphacolor(get(atrcmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
        colgap!(f.layout, 20)
    
        ax = Axis(g1[2,1], xticks=0:0.2:0.4,  xticklabelsize=25, xlabel="1 / yr", xlabelsize=25,titlesize=titlesize, ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
        hidespines!(ax, :l, :t, :r)
        hideydecorations!(ax, label=false)
        CairoMakie.xlims!(ax, 0.0,0.5)
        hist!(vec(Array(bf_pst[:Am_a])), bins=15, color=alphacolor(get(abcmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
        
        ax = Axis(g1[2,2], xticks=0:0.04:0.08, xticklabelsize=25, xlabel="1 / yr", xlabelsize=25,titlesize=titlesize, ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
        hidespines!(ax, :l, :t, :r)
        hideydecorations!(ax, label=false)
        xlims!(ax, 0.0,0.1)
        hist!(vec(Array(bf_pst[:Pm_t])), bins=25, color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
        
        ax = Axis(g1[2,3],  xticks=0:0.05:0.1, xticklabelsize=25, xlabel="1 / yr", xlabelsize=25,titlesize=titlesize, ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
        hidespines!(ax, :l, :t, :r)
        hideydecorations!(ax, label=false)
        xlims!(ax, 0.0,0.125)
        hist!(vec(Array(bf_pst[:Am_t])), bins=15,color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
        
        ax = Axis(g1[2,4], xticks=4:0.5:6, xticklabelsize=25,titlesize=titlesize, ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
        hidespines!(ax, :l, :t, :r)
        hideydecorations!(ax, label=false)
        xlims!(ax, 3.5, 6.5)
        hist!(vec(Array(bf_pst[:β])), bins=8,color=alphacolor(get(taucmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
    
        ax = Axis(g1[2,5], xticks=0.0:0.05:0.2, xticklabelsize=25, xlabel="1 / yr", xlabelsize=25,titlesize=titlesize, ylabelrotation=2pi, ylabelsize=50, ylabelpadding=20)
        hidespines!(ax, :l, :t, :r)
        hideydecorations!(ax, label=false)
        xlims!(ax, 0.0,0.22)
        hist!(vec(Array(bf_pst[:Em])), bins=15, color=alphacolor(get(atrcmap, 0.75), 1.0), strokecolor=:white, strokewidth=1)
        # colgap!(f.layout, 20)

        
        g2 = f[3:4,2] = GridLayout()

        for (i, df) in enumerate([adni_results, bf_results])
                start = 0.6
                stop = 2.0
                border = 0.28
                ax1 = CairoMakie.Axis(g2[i,1],  
                        xlabel="Prediction", 
                        ylabel="Observation", 
                        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                        xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                        xticks=start:0.7:stop, yticks=start:0.7:stop, 
                        xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                        xtickformat = "{:.1f}", ytickformat = "{:.1f}")
                if i == 1
                        hidexdecorations!(ax1, grid=false, ticks=false, )
                end
                CairoMakie.xlims!(ax1, start, stop + border)
                CairoMakie.ylims!(ax1, start, stop + border)
                # lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

                start = 1.0
                stop = 2.0
                border = 0.2
                ax2 =CairoMakie.Axis(g2[i,2],  
                        xlabel="Prediction", 
                        ylabel="Observation", 
                        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                        xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                        xticks=start:0.5:stop, yticks=start:0.5:stop, 
                        xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                        xtickformat = "{:.1f}", ytickformat = "{:.1f}")
                hideydecorations!(ax2, grid=false, ticks=false, ticklabels=false)
                if i == 1
                        hidexdecorations!(ax2, grid=false, ticks=false, )
                end
                CairoMakie.xlims!(ax2, start, stop + border)
                CairoMakie.ylims!(ax2, start, stop + border)
                # lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

                start = 0.0
                stop = 0.2
                border = 0.04
                ax4 =CairoMakie.Axis(g2[i,3],  
                        xlabel="Δ Prediction", 
                        ylabel="Δ Observation", 
                        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                        xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                        xticks=start:0.1:stop, yticks=start:0.1:stop, 
                        xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                        xtickformat = "{:.1f}", ytickformat = "{:.1f}")
                if i == 1
                        hidexdecorations!(ax4, grid=false, ticks=false, )
                end
                CairoMakie.xlims!(ax4, start, stop + border)
                CairoMakie.ylims!(ax4, start, stop + border)
                # lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

                start = -0.0
                stop = 0.6
                border = 0.12
                ax5 =CairoMakie.Axis(g2[i,4],  
                        xlabel="Δ Prediction", 
                        ylabel="Δ Observation", 
                        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                        xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                        xticks=start:0.3:stop, yticks=start:0.3:stop, 
                        xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                        xtickformat = "{:.1f}", ytickformat = "{:.1f}")
                hideydecorations!(ax5, grid=false, ticks=false, ticklabels=false)
                if i == 1
                        hidexdecorations!(ax5, grid=false, ticks=false, )
                end
                CairoMakie.xlims!(ax5, start, stop + border)
                CairoMakie.ylims!(ax5, start, stop + border)
                # lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
                
                start = -0.0
                stop = 0.20
                border = 0.04
                ax3= CairoMakie.Axis(g2[i,5],  
                        xlabel="Δ Prediction", 
                        ylabel="Δ Observation", 
                        titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
                        xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
                        xticks=start:0.1:stop, yticks=start:0.1:stop, 
                        xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
                        xtickformat = "{:.1f}", ytickformat = "{:.1f}")
                hideydecorations!(ax3, grid=false, ticks=false, ticklabels=false)
                if i == 1
                        hidexdecorations!(ax3, grid=false, ticks=false, )
                end
                CairoMakie.xlims!(ax3, start, stop + border)
                CairoMakie.ylims!(ax3, start, stop + border)
                # lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

                if i == 1
                        Label(g2[0, 1], "Aβ", tellwidth=false, fontsize=30)
                        Label(g2[0, 2], "Tau", tellwidth=false, fontsize=30)
                        Label(g2[0, 3], "Aβ", tellwidth=false, fontsize=30)
                        Label(g2[0, 4], "Tau", tellwidth=false, fontsize=30)
                        Label(g2[0, 5], "Neurodeg.", tellwidth=false, fontsize=30)
                end

                labels = ["Braak 1", "Braak 2/3", "Braak 4", "Braak 5", "Braak 6"]
                cmap = reverse(Makie.wong_colors()[1:5])
                #rmse
                abr = round(rmse(df.mean_ab_sols, df.mean_ab), sigdigits=2)
                taur = round(rmse(df.mean_tau_sols, df.mean_tau), sigdigits=2)
                atrr = round(rmse(df.mean_atr_sols, df.mean_atr), sigdigits=2)
                abdr = round(rmse(df.mean_ab_sol_diff, df.mean_ab_diff), sigdigits=2)
                taudr = round(rmse(df.mean_tau_sol_diff, df.mean_tau_diff), sigdigits=2)

                #linear model 
                ablm = lm(@formula( mean_ab ~  mean_ab_sols), df)
                taulm = lm(@formula( mean_tau ~  mean_tau_sols), df)
                atrlm = lm(@formula( mean_atr ~  mean_atr_sols), df)
                abdlm = lm(@formula( mean_ab_diff ~  mean_ab_sol_diff), df)
                taudlm = lm(@formula( mean_tau_diff ~  mean_tau_sol_diff), df)

                testdf = DataFrame(mean_ab_sols = LinRange(0.6, 2.5, 100),
                                   mean_tau_sols = LinRange(1.0, 2.5,100),
                                   mean_atr_sols = LinRange(0.0, 0.5,100),
                                   mean_ab_sol_diff = LinRange(0.0, 0.7,100),
                                   mean_tau_sol_diff = LinRange(0.0, 0.7,100))

                ablm_pred = predict(ablm, testdf, interval=:prediction, level=0.95)
                taulm_pred = predict(taulm, testdf, interval=:prediction, level=0.95)
                atrlm_pred = predict(atrlm, testdf, interval=:prediction, level=0.95)
                abdlm_pred = predict(abdlm, testdf, interval=:prediction, level=0.95)
                taudlm_pred = predict(taudlm, testdf, interval=:prediction, level=0.95)

                band!(ax1,  testdf.mean_ab_sols, disallowmissing(ablm_pred.lower), disallowmissing(ablm_pred.upper), color=(cmap[5], 0.2))
                band!(ax2,  testdf.mean_tau_sols, disallowmissing(taulm_pred.lower), disallowmissing(taulm_pred.upper), color=(cmap[5], 0.2))
                band!(ax4,  testdf.mean_ab_sol_diff, disallowmissing(abdlm_pred.lower), disallowmissing(abdlm_pred.upper), color=(cmap[5], 0.2))
                band!(ax5,  testdf.mean_tau_sol_diff, disallowmissing(taudlm_pred.lower), disallowmissing(taudlm_pred.upper), color=(cmap[5], 0.2))
                band!(ax3,  testdf.mean_atr_sols, disallowmissing(atrlm_pred.lower), disallowmissing(atrlm_pred.upper), color=(cmap[5], 0.2))
                for (i, rois) in enumerate(reverse(bs))                        
                        CairoMakie.scatter!(ax1, df.mean_ab_sols[rois],df.mean_ab[rois], color=cmap[i], markersize=20 ,label=labels[i])
                        CairoMakie.scatter!(ax2, df.mean_tau_sols[rois], df.mean_tau[rois], color=cmap[i], markersize=20, label=labels[i])
                        CairoMakie.scatter!(ax3, df.mean_atr_sols[rois], df.mean_atr[rois], color=cmap[i], markersize=20, label=labels[i])
                        CairoMakie.scatter!(ax4, df.mean_ab_sol_diff[rois], df.mean_ab_diff[rois], color=cmap[i], markersize=20, label=labels[i])
                        CairoMakie.scatter!(ax5, df.mean_tau_sol_diff[rois], df.mean_tau_diff[rois], color=cmap[i], markersize=20, label=labels[i])

                        # CairoMakie.scatter!(ax4, mean_ab_diff[rois], mean_ab_sol_diff[rois], color=cmap[i], markersize=20 , label=labels[i])
                        CairoMakie.text!(ax1, 0.2, 0.85, text= L"\text{RMSE} = %$abr", align=(:left, :bottom), space=:relative, offset=(-40, 0), fontsize=27)
                        CairoMakie.text!(ax1, 0.2, 0.7, text= L"R^{2} = %$(round(r2(ablm), sigdigits=2))", align=(:left, :bottom), space=:relative, offset=(-40, 0), fontsize=27)
                        # CairoMakie.scatter!(ax5, mean_tau_diff[rois], mean_tau_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
                        CairoMakie.text!(ax2, 0.2, 0.85, text= L"\text{RMSE} = %$taur", align=(:left, :bottom), space=:relative, offset=(-40, 0), fontsize=27)
                        CairoMakie.text!(ax2, 0.2, 0.7, text= L"R^{2} = %$(round(r2(taulm), sigdigits=2))", align=(:left, :bottom), space=:relative, offset=(-40, 0), fontsize=27)
                        # CairoMakie.scatter!(ax6, mean_atr_diff[rois], mean_atr_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
                        CairoMakie.text!(ax3, 0.2, 0.85, text= L"\text{RMSE} = %$atrr", align=(:left, :bottom), space=:relative, offset=(-40, 0), fontsize=27)
                        CairoMakie.text!(ax3, 0.2, 0.7, text= L"R^{2} = %$(round(r2(atrlm), sigdigits=2))", align=(:left, :bottom), space=:relative, offset=(-40, 0), fontsize=27)
                        CairoMakie.text!(ax4, 0.2, 0.85, text= L"\text{RMSE} = %$abdr", align=(:left, :bottom), space=:relative, offset=(-40, 0), fontsize=27)
                        CairoMakie.text!(ax4, 0.2, 0.7, text= L"R^{2} = %$(round(r2(abdlm), sigdigits=2))", align=(:left, :bottom), space=:relative, offset=(-40, 0), fontsize=27)
                        CairoMakie.text!(ax5, 0.2, 0.85, text= L"\text{RMSE} = %$taudr", align=(:left, :bottom), space=:relative, offset=(-40, 0), fontsize=27)
                        CairoMakie.text!(ax5, 0.2, 0.7, text= L"R^{2} = %$(round(r2(taudlm), sigdigits=2))", align=(:left, :bottom), space=:relative, offset=(-40, 0), fontsize=27)
                end
                lines!(ax1, testdf.mean_ab_sols, ablm_pred.prediction, color=cmap[5], linewidth=3)
                lines!(ax2, testdf.mean_tau_sols, taulm_pred.prediction, color=cmap[5], linewidth=3)
                lines!(ax4, testdf.mean_ab_sol_diff, abdlm_pred.prediction, color=cmap[5], linewidth=3)
                lines!(ax5, testdf.mean_tau_sol_diff, taudlm_pred.prediction, color=cmap[5], linewidth=3)
                lines!(ax3, testdf.mean_atr_sols, atrlm_pred.prediction, color=cmap[5], linewidth=3)
                if i == 2
                        elems = [[MarkerElement(color = c, marker='●',markersize = 25)] for c in Makie.wong_colors()[1:5]]
                        Legend(g2[3,:], elems, labels, framevisible = false, unique=true, labelsize=35, nbanks=5, tellheight=true, tellwidth=false)
                end
        end     
        rowgap!(f.layout, 50)
        rowsize!(f.layout, 1, 30)
        colsize!(f.layout, 2, 1325)
        Label(g0[1, 1, Top()], "A", fontsize = 40, font = :bold, padding = (50, 0, 0, -70), halign = :center, tellheight=false, tellwidth=false)
        Label(g0[1, 1, ], "ADNI", fontsize = 35, padding = (0, 0, 0, -110), halign = :center, tellheight=false, tellwidth=true, rotation=pi/2)
        Label(g0[2, 1, ], "BF-2", fontsize = 35, padding = (0, 0, 0, -250), halign = :center, tellheight=false, tellwidth=true, rotation=pi/2)
        Label(g0[3, 1, Top()], "B", fontsize = 40, font = :bold, padding = (50, 0, 0, -120), halign = :center, tellheight=false, tellwidth=false)
        Label(g0[3, 1, ], "ADNI", fontsize = 35, padding = (0, 0, 0, -110), halign = :center, tellheight=false, tellwidth=true, rotation=pi/2)
        Label(g0[4, 1, ], "BF-2", fontsize = 35, padding = (0, 0, 0, -225), halign = :center, tellheight=false, tellwidth=true, rotation=pi/2)
        f
end
save(projectdir("output/plots/inference/pst-pstpred-harmonised-suvr-adni-bf-random-lognormal.png"),f)

# begin
#     cmap = Makie.wong_colors();

#     f = Figure(size=(1500, 900))
#     titlesize = 40
#     xlabelsize = 25 
#     ylabelsize = 25
#     xticklabelsize = 25 
#     yticklabelsize = 25
#     rsize=30

#     start = 0.2
#     stop = 1.0
#     border = 0.03
#     Label(f[0, 1], "Aβ", tellwidth=false, fontsize=35)
#     Label(f[0, 2], "Tau", tellwidth=false, fontsize=35)
#     Label(f[0, 3], "Neurodegeneration", tellwidth=false, fontsize=35)
#     Label(f[1, 0], "ADNI", tellwidth=true, tellheight=false, fontsize=35)
#     Label(f[2, 0], "BF-2", tellwidth=true, tellheight=false, fontsize=35)

#     ax1 =CairoMakie.Axis(f[1,1],  
#             xlabel="Observation", 
#             ylabel="Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.2:stop, yticks=start:0.2:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     hidexdecorations!(ax1, grid=false, ticks=false, )
#     CairoMakie.xlims!(ax1, start, stop + border)
#     CairoMakie.ylims!(ax1, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
#     start = -0.0
#     stop = 0.6
#     border = 0.03
#     ax2 =CairoMakie.Axis(f[1,2],  
#             xlabel="Observation", 
#             ylabel="Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.15:stop, yticks=start:0.15:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     hideydecorations!(ax2, grid=false, ticks=false, ticklabels=false)
#     hidexdecorations!(ax2, grid=false, ticks=false, )
#     CairoMakie.xlims!(ax2, start, stop + border)
#     CairoMakie.ylims!(ax2, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

#     start = -0.0
#     stop = 0.2
#     border = 0.01
#     ax3= CairoMakie.Axis(f[1,3],  
#             xlabel="Observation", 
#             ylabel="Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.05:stop, yticks=start:0.05:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     hideydecorations!(ax3, grid=false, ticks=false, ticklabels=false)
#     hidexdecorations!(ax3, grid=false, ticks=false, )
#     CairoMakie.xlims!(ax3, start, stop + border)
#     CairoMakie.ylims!(ax3, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
#     start = 0.2
#     stop = 1.0
#     border = 0.03

#     ax4 =CairoMakie.Axis(f[2,1],  
#             xlabel="Observation", 
#             ylabel="Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.2:stop, yticks=start:0.2:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")

#     CairoMakie.xlims!(ax4, start, stop + border)
#     CairoMakie.ylims!(ax4, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
#     start = -0.0
#     stop = 0.6
#     border = 0.03
#     ax5 =CairoMakie.Axis(f[2,2],  
#             xlabel="Observation", 
#             ylabel="Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.2:stop, yticks=start:0.2:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     hideydecorations!(ax5, grid=false, ticks=false, ticklabels=false)
#     CairoMakie.xlims!(ax5, start, stop + border)
#     CairoMakie.ylims!(ax5, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

#     start = -0.0
#     stop = 0.2
#     border = 0.01
#     ax6 = CairoMakie.Axis(f[2,3],  
#             xlabel="Observation", 
#             ylabel="Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.05:stop, yticks=start:0.05:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     hideydecorations!(ax6, grid=false, ticks=false, ticklabels=false)
#     CairoMakie.xlims!(ax6, start, stop + border)
#     CairoMakie.ylims!(ax6, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    

#     ab_sols = Vector{Array{Float64}}()
#     tau_sols = Vector{Array{Float64}}()
#     atr_sols = Vector{Array{Float64}}()
#     meanpst = mean(pst)

#     for sub in 1:22
#         p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
#         meanpst[Symbol("α_t[$sub]"), :mean], meanpst[Symbol("β"), :mean], 
#         meanpst[Symbol("η[$sub]"), :mean]]
        
#         pstprob = remake(fbb_prob, u0=fbb_inits[sub], p=p)
#         pstsol = solve(pstprob, Tsit5())
#         push!(ab_sols,pstsol(fbb_times[sub])[1:72,:])
#         push!(tau_sols,pstsol(fbb_tau_times[sub])[73:144,:])
#         push!(atr_sols,pstsol(fbb_tau_times[sub])[145:216,:])
#     end

#     for (i, sub) in enumerate(23:44)
#         p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
#         meanpst[Symbol("α_t[$sub]"), :mean], meanpst[Symbol("β"), :mean], 
#         meanpst[Symbol("η[$sub]"), :mean]]
        
#         pstprob = remake(fbp_prob, u0=fbp_inits[i], p=p)
#         pstsol = solve(pstprob, Tsit5())
#         push!(ab_sols,  pstsol(fbp_times[i])[1:72,:])
#         push!(tau_sols, pstsol(fbp_tau_times[i])[73:144,:])
#         push!(atr_sols, pstsol(fbp_tau_times[i])[145:216,:])
#     end

#     mean_ab_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in ab_sols]), dims=2))
#     mean_tau_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in tau_sols]), dims=2))
#     mean_atr_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in atr_sols]), dims=2))
    
#     mean_ab =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_conc; fbp_conc]]), dims=2))
#     mean_tau =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_tau_conc; fbp_tau_conc]]), dims=2))
#     mean_atr =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_vols; fbp_vols]]), dims=2))

#     # mean_ab =  vec(mean(reduce(hcat, [d[:, end] for d in fbb_conc]), dims=2)) #; fbp_conc]]), dims=2))
#     # mean_tau =  vec(mean(reduce(hcat, [d[:, end] for d in fbb_tau_conc]), dims=2)) #; fbp_tau_conc]]), dims=2))
#     # mean_atr =  vec(mean(reduce(hcat, [d[:, end] for d in fbb_vols]), dims=2)) #; fbp_vols]]), dims=2))
    
#     mean_ab_sol_diff = vec(mean(reduce(hcat, get_diff.(ab_sols)), dims=2))
#     mean_tau_sol_diff = vec(mean(reduce(hcat, get_diff.(tau_sols)), dims=2))
#     mean_atr_sol_diff = vec(mean(reduce(hcat, get_diff.(atr_sols)), dims=2))

#     mean_ab_diff = vec(mean(reduce(hcat, get_diff.([fbb_conc; fbp_conc])), dims=2))
#     mean_tau_diff = vec(mean(reduce(hcat, get_diff.([fbb_tau_conc; fbp_tau_conc])), dims=2))
#     mean_atr_diff = vec(mean(reduce(hcat, get_diff.([fbb_vols; fbp_vols])), dims=2))

#     # mean_ab_diff = vec(mean(reduce(hcat, get_diff.(fbb_conc)), dims=2)) #; fbp_conc])), dims=2))
#     # mean_tau_diff = vec(mean(reduce(hcat, get_diff.(fbb_tau_conc)), dims=2)) #; fbp_tau_conc])), dims=2))
#     # mean_atr_diff = vec(mean(reduce(hcat, get_diff.(fbb_vols)), dims=2)) #; fbp_vols])), dims=2))
#     abr = round(rsquared(mean_ab_sols, mean_ab), sigdigits=2)
#     taur = round(rsquared(mean_tau_sols, mean_tau), sigdigits=2)
#     atrr = round(rsquared(mean_atr_sols, mean_atr), sigdigits=2)

#     bf_abr = round(rsquared(bf_results.mean_ab_sols, bf_results.mean_ab), sigdigits=2)
#     bf_taur = round(rsquared(bf_results.mean_tau_sols, bf_results.mean_tau), sigdigits=2)
#     bf_atrr = round(rsquared(bf_results.mean_atr_sols, bf_results.mean_atr), sigdigits=2)

#     # abr_diff = round(rsquared(mean_ab_sol_diff, mean_ab_diff), sigdigits=2)
#     # taur_diff = round(rsquared(mean_tau_sol_diff, mean_tau_diff), sigdigits=2)
#     # atrr_diff = round(rsquared(mean_atr_sol_diff, mean_atr_diff), sigdigits=2)
#     labels = ["Braak 1", "Braak 2/3", "Braak 4", "Braak 5", "Braak 6"]
    
#     for (i, rois) in enumerate(bs)
#         CairoMakie.scatter!(ax1, mean_ab[rois], mean_ab_sols[rois], color=cmap[i], markersize=20 , label=labels[i])
#         CairoMakie.text!(ax1, 1.0, 0., text= L"R^{2} = %$abr", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=rsize)
#         CairoMakie.scatter!(ax2, mean_tau[rois], mean_tau_sols[rois], color=cmap[i], markersize=20, label=labels[i])
#         CairoMakie.text!(ax2, 1.0, 0., text= L"R^{2} = %$taur", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=rsize)
#         CairoMakie.scatter!(ax3, mean_atr[rois], mean_atr_sols[rois], color=cmap[i], markersize=20, label=labels[i])
#         CairoMakie.text!(ax3, 1.0, 0., text= L"R^{2} = %$atrr", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=rsize)

#         CairoMakie.scatter!(ax4, bf_results.mean_ab[rois], bf_results.mean_ab_sols[rois], color=cmap[i], markersize=20 , label=labels[i])
#         CairoMakie.text!(ax4, 1.0, 0., text= L"R^{2} = %$bf_abr", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=rsize)
#         CairoMakie.scatter!(ax5, bf_results.mean_tau[rois], bf_results.mean_tau_sols[rois], color=cmap[i], markersize=20, label=labels[i])
#         CairoMakie.text!(ax5, 1.0, 0., text= L"R^{2} = %$bf_taur", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=rsize)
#         CairoMakie.scatter!(ax6, bf_results.mean_atr[rois], bf_results.mean_atr_sols[rois], color=cmap[i], markersize=20, label=labels[i])
#         CairoMakie.text!(ax6, 1.0, 0., text= L"R^{2} = %$bf_atrr", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=rsize)

#         # CairoMakie.scatter!(ax4, mean_ab_diff[rois], mean_ab_sol_diff[rois], color=cmap[i], markersize=20 , label=labels[i])
#         # CairoMakie.text!(ax4, 1.0, 0., text= L"R^{2} = %$abr_diff", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=25)
#         # CairoMakie.scatter!(ax5, mean_tau_diff[rois], mean_tau_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
#         # CairoMakie.text!(ax5, 1.0, 0., text= L"R^{2} = %$taur_diff", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=25)
#         # CairoMakie.scatter!(ax6, mean_atr_diff[rois], mean_atr_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
#         # CairoMakie.text!(ax6, 1.0, 0., text= L"R^{2} = %$atrr_diff", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=25)
#     end
#     Legend(f[3,:], ax1, framevisible = false, unique=true, labelsize=35, nbanks=5, tellheight=true, tellwidth=false)

#     f
# end
# save(projectdir("output/plots/inference-results/pst-pred-harmonised-scaled-adni-bf.pdf"),f)

# begin
#     cmap = Makie.wong_colors();

#     f = Figure(size=(1500, 500))
#     titlesize = 40
#     xlabelsize = 25 
#     ylabelsize = 25
#     xticklabelsize = 20 
#     yticklabelsize = 20

#     start = -0.0
#     stop = 0.2
#     border = 0.01
#     ax4 =CairoMakie.Axis(f[1,1],  
#             xlabel="Δ SUVR", 
#             ylabel="Δ Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.05:stop, yticks=start:0.05:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")

#     CairoMakie.xlims!(ax4, start, stop + border)
#     CairoMakie.ylims!(ax4, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
#     start = -0.0
#     stop = 0.2
#     border = 0.01
#     ax5 =CairoMakie.Axis(f[1,2],  
#             xlabel="Δ SUVR", 
#             ylabel="Δ Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.05:stop, yticks=start:0.05:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     hideydecorations!(ax5, grid=false, ticks=false, ticklabels=false)
#     CairoMakie.xlims!(ax5, start, stop + border)
#     CairoMakie.ylims!(ax5, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

#     start = -0.0
#     stop = 0.2
#     border = 0.01
#     ax6= CairoMakie.Axis(f[1,3],  
#             xlabel="Δ Atr.", 
#             ylabel="Δ Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.05:stop, yticks=start:0.05:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     hideydecorations!(ax6, grid=false, ticks=false, ticklabels=false)
#     CairoMakie.xlims!(ax6, start, stop + border)
#     CairoMakie.ylims!(ax6, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
#     ab_sols = Vector{Array{Float64}}()
#     tau_sols = Vector{Array{Float64}}()
#     atr_sols = Vector{Array{Float64}}()
#     meanpst = mean(pst)

#     for sub in 1:22
#         p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
#         meanpst[Symbol("α_t[$sub]"), :mean], meanpst[Symbol("β"), :mean], 
#         meanpst[Symbol("η[$sub]"), :mean]]
        
#         pstprob = remake(fbb_prob, u0=fbb_inits[sub], p=p)
#         pstsol = solve(pstprob, Tsit5())
#         push!(ab_sols,pstsol(fbb_times[sub])[1:72,:])
#         push!(tau_sols,pstsol(fbb_tau_times[sub])[73:144,:])
#         push!(atr_sols,pstsol(fbb_tau_times[sub])[145:216,:])
#     end

#     for (i, sub) in enumerate(23:44)
#         p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
#         meanpst[Symbol("α_t[$sub]"), :mean], meanpst[Symbol("β"), :mean], 
#         meanpst[Symbol("η[$sub]"), :mean]]
        
#         pstprob = remake(fbp_prob, u0=fbp_inits[i], p=p)
#         pstsol = solve(pstprob, Tsit5())
#         push!(ab_sols,  pstsol(fbp_times[i])[1:72,:])
#         push!(tau_sols, pstsol(fbp_tau_times[i])[73:144,:])
#         push!(atr_sols, pstsol(fbp_tau_times[i])[145:216,:])
#     end

#     mean_ab_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in ab_sols]), dims=2))
#     mean_tau_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in tau_sols]), dims=2))
#     mean_atr_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in atr_sols]), dims=2))
    
#     mean_ab =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_conc; fbp_conc]]), dims=2))
#     mean_tau =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_tau_conc; fbp_tau_conc]]), dims=2))
#     mean_atr =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_vols; fbp_vols]]), dims=2))

#     # mean_ab =  vec(mean(reduce(hcat, [d[:, end] for d in fbb_conc]), dims=2)) #; fbp_conc]]), dims=2))
#     # mean_tau =  vec(mean(reduce(hcat, [d[:, end] for d in fbb_tau_conc]), dims=2)) #; fbp_tau_conc]]), dims=2))
#     # mean_atr =  vec(mean(reduce(hcat, [d[:, end] for d in fbb_vols]), dims=2)) #; fbp_vols]]), dims=2))
    
#     mean_ab_sol_diff = vec(mean(reduce(hcat, get_diff.(ab_sols)), dims=2))
#     mean_tau_sol_diff = vec(mean(reduce(hcat, get_diff.(tau_sols)), dims=2))
#     mean_atr_sol_diff = vec(mean(reduce(hcat, get_diff.(atr_sols)), dims=2))

#     mean_ab_diff = vec(mean(reduce(hcat, get_diff.([fbb_conc; fbp_conc])), dims=2))
#     mean_tau_diff = vec(mean(reduce(hcat, get_diff.([fbb_tau_conc; fbp_tau_conc])), dims=2))
#     mean_atr_diff = vec(mean(reduce(hcat, get_diff.([fbb_vols; fbp_vols])), dims=2))

#     # mean_ab_diff = vec(mean(reduce(hcat, get_diff.(fbb_conc)), dims=2)) #; fbp_conc])), dims=2))
#     # mean_tau_diff = vec(mean(reduce(hcat, get_diff.(fbb_tau_conc)), dims=2)) #; fbp_tau_conc])), dims=2))
#     # mean_atr_diff = vec(mean(reduce(hcat, get_diff.(fbb_vols)), dims=2)) #; fbp_vols])), dims=2))

#     labels = ["Braak 1", "Braak 2/3", "Braak 4", "Braak 5", "Braak 6"]
#     for (i, rois) in enumerate(bs)
#         CairoMakie.scatter!(ax4, mean_ab_diff[rois], mean_ab_sol_diff[rois], color=cmap[i], markersize=20 , label=labels[i])
#         CairoMakie.scatter!(ax5, mean_tau_diff[rois], mean_tau_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
#         CairoMakie.scatter!(ax6, mean_atr_diff[rois], mean_atr_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
#     end
#     Legend(f[2,:], ax6, framevisible = false, unique=true, labelsize=35, nbanks=5, tellheight=true)

#     f
# end
# save(projectdir("output/plots/inference-results/pst-pred-harmonised-scaled.pdf"),f)

# begin
#     cmap = Makie.wong_colors();

#     f = Figure(size=(1500, 900))
#     titlesize = 40
#     xlabelsize = 25 
#     ylabelsize = 25
#     xticklabelsize = 25 
#     yticklabelsize = 25

#     start = 0.4
#     stop = 1.0
#     border = 0.03
#     Label(f[3, 1], "Aβ", tellwidth=false, fontsize=35)
#     Label(f[3, 2], "Tau", tellwidth=false, fontsize=35)
#     Label(f[3, 3], "Neurodegeneration", tellwidth=false, fontsize=35)
#     ax1 =CairoMakie.Axis(f[1,1],  
#             xlabel="Observation", 
#             ylabel="Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.2:stop, yticks=start:0.2:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")

#     CairoMakie.xlims!(ax1, start, stop + border)
#     CairoMakie.ylims!(ax1, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
#     start = -0.0
#     stop = 0.6
#     border = 0.03
#     ax2 =CairoMakie.Axis(f[1,2],  
#             xlabel="Observation", 
#             ylabel="Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.2:stop, yticks=start:0.2:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     hideydecorations!(ax2, grid=false, ticks=false, ticklabels=false)
#     CairoMakie.xlims!(ax2, start, stop + border)
#     CairoMakie.ylims!(ax2, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

#     # start = -0.0
#     # stop = 0.2
#     # border = 0.01
#     # ax3= CairoMakie.Axis(f[1,3],  
#     #         xlabel="Observation", 
#     #         ylabel="Prediction", 
#     #         titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#     #         xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#     #         xticks=start:0.05:stop, yticks=start:0.05:stop, 
#     #         xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#     #         xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     # hideydecorations!(ax3, grid=false, ticks=false, ticklabels=false)
#     # CairoMakie.xlims!(ax3, start, stop + border)
#     # CairoMakie.ylims!(ax3, start, stop + border)
#     # lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

#     start = -0.0
#     stop = 0.2
#     border = 0.01
#     ax4 =CairoMakie.Axis(f[2,1],  
#             xlabel="Δ Observation", 
#             ylabel="Δ Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.05:stop, yticks=start:0.05:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")

#     CairoMakie.xlims!(ax4, start, stop + border)
#     CairoMakie.ylims!(ax4, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
#     start = -0.0
#     stop = 0.2
#     border = 0.01
#     ax5 =CairoMakie.Axis(f[2,2],  
#             xlabel="Δ Observation", 
#             ylabel="Δ Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.05:stop, yticks=start:0.05:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     hideydecorations!(ax5, grid=false, ticks=false, ticklabels=false)
#     CairoMakie.xlims!(ax5, start, stop + border)
#     CairoMakie.ylims!(ax5, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

#     start = -0.0
#     stop = 0.2
#     border = 0.01
#     ax6= CairoMakie.Axis(f[2,3],  
#             xlabel="Δ Observation.", 
#             ylabel="Δ Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.05:stop, yticks=start:0.05:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     hideydecorations!(ax6, grid=false, ticks=false, ticklabels=false)
#     CairoMakie.xlims!(ax6, start, stop + border)
#     CairoMakie.ylims!(ax6, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
#     ab_sols = Vector{Array{Float64}}()
#     tau_sols = Vector{Array{Float64}}()
#     atr_sols = Vector{Array{Float64}}()
#     meanpst = mean(pst)

#     for sub in 1:22
#         p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
#         meanpst[Symbol("α_t[$sub]"), :mean], meanpst[Symbol("β"), :mean], 
#         meanpst[Symbol("η[$sub]"), :mean]]
        
#         pstprob = remake(fbb_prob, u0=fbb_inits[sub], p=p)
#         pstsol = solve(pstprob, Tsit5())
#         push!(ab_sols,pstsol(fbb_times[sub])[1:72,:])
#         push!(tau_sols,pstsol(fbb_tau_times[sub])[73:144,:])
#         push!(atr_sols,pstsol(fbb_tau_times[sub])[145:216,:])
#     end

#     for (i, sub) in enumerate(23:44)
#         p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
#         meanpst[Symbol("α_t[$sub]"), :mean], meanpst[Symbol("β"), :mean], 
#         meanpst[Symbol("η[$sub]"), :mean]]
        
#         pstprob = remake(fbp_prob, u0=fbp_inits[i], p=p)
#         pstsol = solve(pstprob, Tsit5())
#         push!(ab_sols,  pstsol(fbp_times[i])[1:72,:])
#         push!(tau_sols, pstsol(fbp_tau_times[i])[73:144,:])
#         push!(atr_sols, pstsol(fbp_tau_times[i])[145:216,:])
#     end

#     mean_ab_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in ab_sols]), dims=2))
#     mean_tau_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in tau_sols]), dims=2))
#     mean_atr_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in atr_sols]), dims=2))
    
#     mean_ab =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_conc; fbp_conc]]), dims=2))
#     mean_tau =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_tau_conc; fbp_tau_conc]]), dims=2))
#     mean_atr =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_vols; fbp_vols]]), dims=2))

#     # mean_ab =  vec(mean(reduce(hcat, [d[:, end] for d in fbb_conc]), dims=2)) #; fbp_conc]]), dims=2))
#     # mean_tau =  vec(mean(reduce(hcat, [d[:, end] for d in fbb_tau_conc]), dims=2)) #; fbp_tau_conc]]), dims=2))
#     # mean_atr =  vec(mean(reduce(hcat, [d[:, end] for d in fbb_vols]), dims=2)) #; fbp_vols]]), dims=2))
    
#     mean_ab_sol_diff = vec(mean(reduce(hcat, get_diff.(ab_sols)), dims=2))
#     mean_tau_sol_diff = vec(mean(reduce(hcat, get_diff.(tau_sols)), dims=2))
#     mean_atr_sol_diff = vec(mean(reduce(hcat, get_diff.(atr_sols)), dims=2))

#     mean_ab_diff = vec(mean(reduce(hcat, get_diff.([fbb_conc; fbp_conc])), dims=2))
#     mean_tau_diff = vec(mean(reduce(hcat, get_diff.([fbb_tau_conc; fbp_tau_conc])), dims=2))
#     mean_atr_diff = vec(mean(reduce(hcat, get_diff.([fbb_vols; fbp_vols])), dims=2))

#     # mean_ab_diff = vec(mean(reduce(hcat, get_diff.(fbb_conc)), dims=2)) #; fbp_conc])), dims=2))
#     # mean_tau_diff = vec(mean(reduce(hcat, get_diff.(fbb_tau_conc)), dims=2)) #; fbp_tau_conc])), dims=2))
#     # mean_atr_diff = vec(mean(reduce(hcat, get_diff.(fbb_vols)), dims=2)) #; fbp_vols])), dims=2))
#     abr = round(rsquared(mean_ab_sols, mean_ab), sigdigits=2)
#     taur = round(rsquared(mean_tau_sols, mean_tau), sigdigits=2)
#     atrr = round(rsquared(mean_atr_sols, mean_atr), sigdigits=2)

#     abr_diff = round(rsquared(mean_ab_sol_diff, mean_ab_diff), sigdigits=2)
#     taur_diff = round(rsquared(mean_tau_sol_diff, mean_tau_diff), sigdigits=2)
#     atrr_diff = round(rsquared(mean_atr_sol_diff, mean_atr_diff), sigdigits=2)
#     labels = ["Braak 1", "Braak 2/3", "Braak 4", "Braak 5", "Braak 6"]
    
#     for (i, rois) in enumerate(bs)
#         CairoMakie.scatter!(ax1, mean_ab[rois], mean_ab_sols[rois], color=cmap[i], markersize=20 , label=labels[i])
#         CairoMakie.text!(ax1, 1.0, 0., text= L"R^{2} = %$abr", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=25)
#         CairoMakie.scatter!(ax2, mean_tau[rois], mean_tau_sols[rois], color=cmap[i], markersize=20, label=labels[i])
#         CairoMakie.text!(ax2, 1.0, 0., text= L"R^{2} = %$taur", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=25)
#         # CairoMakie.scatter!(ax3, mean_atr[rois], mean_atr_sols[rois], color=cmap[i], markersize=20, label=labels[i])
#         # CairoMakie.text!(ax3, 1.0, 0., text= L"R^{2} = %$atrr", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=25)

#         CairoMakie.scatter!(ax4, mean_ab_diff[rois], mean_ab_sol_diff[rois], color=cmap[i], markersize=20 , label=labels[i])
#         CairoMakie.text!(ax4, 1.0, 0., text= L"R^{2} = %$abr_diff", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=25)
#         CairoMakie.scatter!(ax5, mean_tau_diff[rois], mean_tau_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
#         CairoMakie.text!(ax5, 1.0, 0., text= L"R^{2} = %$taur_diff", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=25)
#         CairoMakie.scatter!(ax6, mean_atr_diff[rois], mean_atr_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
#         CairoMakie.text!(ax6, 1.0, 0., text= L"R^{2} = %$atrr_diff", align=(:right, :bottom), space=:relative, offset=(-20, 10), fontsize=25)
#     end
#     Legend(f[1,3], ax5, framevisible = false, unique=true, labelsize=35, nbanks=2, tellheight=false, tellwidth=false)

#     f
# end
# save(projectdir("output/plots/inference-results/pst-pred-harmonised-scaled-all.pdf"),f)

# pst = deserialize(projectdir("output/chains/population-scaled-atn/pst-samples-harmonised-ind-diag-1x1000.jls"));
# summarize(pst)
# meanpst = mean(pst);

# using CairoMakie; CairoMakie.activate!()

# mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
# mtl = findall(x -> x ∈ mtl_regions, get_label.(cortex)) 
# neo_regions = ["inferiortemporal", "middletemporal"] 
# neo = findall(x -> x ∈ neo_regions, get_label.(cortex))

# rois = [mtl ; neo]
# ms = [mean(pst["β[$i]"]) for i in 1:44]
# gms = [mean(pst["α_t[$i]"]) for i in 1:44]

# cs =[cov(vec(pst["β[$i]"]), vec(pst["α_t[$i]"])) for i in 1:44]
# stds = [std(pst["β[$i]"]) for i in 1:44]
# mean_tau_inits = [[mean(fbb_tau_inits[i][rois]) for i in 1:22]; [mean(fbp_tau_inits[i][rois]) for i in 1:22]]
# mean_ab_inits = [[mean(fbb_inits[i][rois]) for i in 1:22]; [mean(fbp_inits[i][rois]) for i in 1:22]]
# total_tau =  [[sum(fbb_tau_inits[i]) for i in 1:22]; [sum(fbp_tau_inits[i]) for i in 1:22]]
# tota_ab = [[sum(fbb_inits[i]) for i in 1:22]; [sum(fbp_inits[i]) for i in 1:22]]

# scatter(ms, gms)
# scatter(cs, stds)
# scatter(ms[1:22], cs[1:22])
# scatter(ms[23:44], cs[23:44])
# scatter(ms, stds)
# scatter(ms, mean_tau_inits)
# scatter(ms, mean_ab_inits .* mean_tau_inits)
# scatter(stds, mean_ab_inits .* mean_tau_inits)
# scatter(stds, mean_ab_inits)
# scatter(stds, mean_tau_inits)

# using GLMakie; GLMakie.activate!()
# begin
#     f = Figure(size=(500, 600))
#     ax = Axis3(f[1,1], xlabel="ms", ylabel="stds", zlabel="tau inits")
#     # scatter!(ms[1:22], stds[1:22], (mean_tau_inits .* mean_ab_inits)[1:22], markersize=10, color=:blue)
#     # scatter!(ms[23:44], stds[23:44], (mean_tau_inits .* mean_ab_inits)[23:44], markersize=10, color=:red)
#     scatter(ms, gms, cs)
#     f
# end

# fmm_u0, fmm_ui = load_ab_params(tracer="FMM")
# bf_v0, bf_vi, bf_part = load_tau_params(tracer="RO")
# using CairoMakie
# begin
#     f = Figure(size=(600, 500))
#     ax = Axis(f[1,1], xlabel="Amyloid", ylabel="Tau")
#     CairoMakie.ylims!(ax, 0.0, 5.0)
#     CairoMakie.xlims!(ax, 0.0, 5.0)
#     CairoMakie.scatter!(part .+ 4.5 .* (fbb_ui .- fbb_u0), vi .+ v0)    
#     lines!(collect(0:0.1:5), collect(0:0.1:5), linestyle=:dash)
#     f
# end

# begin
#         f = Figure(size=(600, 500))
#         ax = Axis(f[1,1], xlabel="Amyloid", ylabel="Tau")
#         CairoMakie.ylims!(ax, 0.0, 8.0)
#         CairoMakie.xlims!(ax, 0.0, 8.0)
#         CairoMakie.scatter!(bf_part .+ 6. .* (fmm_ui .- fmm_u0), bf_vi .+ bf_v0)    
#         f
# end
