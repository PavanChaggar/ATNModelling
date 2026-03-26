using ATNModelling.SimulationUtils: load_ab_params, load_tau_params, conc, make_scaled_atn_pkpd_model
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names, get_distance_laplacian, get_braak_regions,
                                    get_subcortex
using ATNModelling.DataUtils: normalise!

using Connectomes: laplacian_matrix, get_hemisphere
using ADNIDatasets: ADNIDataset, get_id, calc_suvr
using DrWatson: projectdir, datadir
using CSV, DataFrames
using CairoMakie, Colors, ColorSchemes, GLMakie
using Statistics, SciMLBase
using LinearAlgebra
using DelimitedFiles
using Turing
using Serialization
# --------------------------------------------------------------------------------
# Tracer independent data
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params()
parc = get_parcellation() |> get_cortex
subcortical_parc = get_parcellation(;) |> get_subcortex
cortex = filter(x -> get_hemisphere(x) == "left", parc)
subcortex = filter(x -> get_hemisphere(x) == "left", subcortical_parc)

tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)

c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
Ld = get_distance_laplacian()
L = laplacian_matrix(c) 

cingulate = findall(x -> contains(get_label(x), "cingulate"), cortex)
m = zeros(36)
m[cingulate] .= 1
m[[35,36]] .= 1

dktnames = get_parcellation() |> get_cortex |> get_dkt_names

pst = chainscat([deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-random-beta-lognormal-4x1000-$i.jls")) for i in 1:4]...)
meanpst = mean(pst)


_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 
tau_cutoffs = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-1std.csv")) |> vec
ab_threshold = readdlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/ab-thresholds.csv")) |> vec
tau_threshold = readdlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/tau-thresholds.csv")) |> vec

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 && x.NEO_Status == 0, tau_data_df);
_tau_data = ADNIDataset(tau_data_df, dktnames; min_scans=1)
tau_subs = get_id.(_tau_data)

amyloid_production = mean([meanpst["α_a[$i]", :mean] for i in 1:18]) / 12
tau_transport = mean([meanpst["ρ_t[$i]", :mean] for i in 1:18]) / 12
tau_production = mean([meanpst["α_t[$i]", :mean] for i in 1:18]) / 12
coupling = meanpst["β_fbb", :mean]
atrophy = mean([meanpst["η[$i]", :mean] for i in 1:18]) / 12

drug_concentration = 400.
drug_transport = 1.5 / 12
drug_effect = 0.12 / 12
drug_clearance = 6. / 12

L = laplacian_matrix(c) 
Δ = part .+ (coupling .* (fbb_ui .- fbb_u0))
Lh = inv(diagm(vi .- v0)) * L * diagm(vi .- v0)

bs = get_braak_regions()
rbs = [filter(x -> x > 42, b) for b in bs]
b3 = reduce(vcat, rbs[1:3])
rois = findall(x -> get_node_id(x) ∈ b3, cortex)
dktnames[rois]

ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);
abcmap = ColorScheme(ab_c);
taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
atrcmap = ColorScheme(atr_c); #ColorSchemes.Reds;

# --------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------
begin
    CairoMakie.activate!()
    f = Figure(size=(800, 1200), figure_padding=(20,20,20,20))
  
    for (i, j) in enumerate(60:10:100)
    println(j)
    # fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer && x.AMYLOID_STATUS_COMPOSITE_REF == 1 && x.RID ∈ tau_subs && x.CENTILOIDS < 67, _ab_data_df);
    fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer 
                            && x.AMYLOID_STATUS_COMPOSITE_REF == 1 && x.RID ∈ tau_subs 
                            && x.CENTILOIDS < j, _ab_data_df);
                            #   && x.CENTILOIDS < 60, _ab_data_df);
    mean(fbb_data_df.CENTILOIDS)
    amy_pos_init_idx = [findfirst(isequal(id), _ab_data_df.RID) for id in unique(fbb_data_df.RID)]
    pos_centiloids = mean(_ab_data_df[amy_pos_init_idx, :].CENTILOIDS)
    pos_centiloids_st = std(_ab_data_df[amy_pos_init_idx, :].CENTILOIDS)

    fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF")

    tau_data = filter(x -> get_id(x) ∈ get_id.(fbb_data), _tau_data)
    CSV.write(projectdir("data/ADNI/pkpd-subs.csv"), DataFrame(sub_id = get_id.(tau_data)))

    fbb_suvr = calc_suvr.(fbb_data)
    normalise!(fbb_suvr, fbb_u0, fbb_ui)
    fbb_conc = map(x -> conc.(x, fbb_u0, fbb_ui), fbb_suvr)
    fbb_inits = [d[:,1] for d in fbb_conc]
    mean_fbb_init = mean(fbb_inits)[1:36]
    println("ab concentration = $(mean_fbb_init[29])")

    fbb_tau_suvr = calc_suvr.(tau_data)
    vi = part .+ (meanpst["β_fbb",:mean] .* (fbb_ui .- fbb_u0))
    # vi = part .+ (4.5.* (fbb_ui .- fbb_u0))
    normalise!(fbb_tau_suvr, v0, vi)
    fbb_tau_conc = map(x -> conc.(x, v0, vi), fbb_tau_suvr)
    fbb_tau_inits = [d[:,1] for d in fbb_tau_conc]

    _mean_tau_init = mean(fbb_tau_inits)
    idx = _mean_tau_init .< conc.(tau_cutoffs, v0, vi)
    _mean_tau_init[idx] .= 0
    mean_tau_init = mean.(zip(_mean_tau_init[1:36], _mean_tau_init[37:end]))
    scatter(mean_tau_init)
    vol_init = zeros(36)
    tmax = 360
    ts = range(0, tmax, tmax * 10)

    atn_pkpd = make_scaled_atn_pkpd_model(fbb_ui[37:72] .- fbb_u0[37:72], part[37:72] .- v0[37:72], L[37:72, 37:72], Ld[37:72, 37:72], m, 0)

    sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                    (0.0, tmax), [amyloid_production, tau_transport, tau_production, 
                                            coupling, atrophy, 
                                            drug_transport, drug_effect, 
                                            drug_concentration, drug_clearance]; 
                                            saveat=ts, tol=1e-12)

    solts = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                    (0.0, tmax), [amyloid_production, tau_transport, tau_production, 
                            coupling, atrophy, 
                            drug_transport, drug_effect, 
                            drug_concentration, drug_clearance]; 
                            saveat=ts, tol=1e-12)

    placebo_sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                    (0.0, tmax), [amyloid_production, tau_transport, tau_production, 
                                            coupling, atrophy, 
                                            drug_transport, 0.0, 
                                            drug_concentration, drug_clearance]; 
                                            saveat=ts, tol=1e-12)

    placebo_solts = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                    (0.0, tmax), [amyloid_production, tau_transport, tau_production, 
                            coupling, atrophy, 
                            drug_transport, 0.0, 
                            drug_concentration, drug_clearance]; 
                            saveat=ts, tol=1e-12)

    absol = Array(placebo_sol[1:36,:])
    tausol = Array(placebo_sol[37:72,:])
    atrsol = Array(placebo_sol[73:108,:])
    drugsol = Array(placebo_sol[109:144,:]) 

    tau_seed_idx = tausol .>= tau_threshold[1:36];
    ab_threshold=fill(0.5, 36)
    ab_seed_idx = absol .>= ab_threshold[1:36];

    ab_tau_coloc = tau_seed_idx .* ab_seed_idx
    coloc_t = findall(x -> x > 0, sum(ab_tau_coloc, dims=1))
    coloc_node = findall(x -> x > 0, ab_tau_coloc[:, coloc_t[1][2]])
    dktnames[coloc_node][1]
    sol.t[coloc_t[1][2]]
    tau_t = sol.t[coloc_t[1][2]]


    sols = Vector{ODESolution}()
    absols = Vector{Array{Float64}}()
    tausols = Vector{Array{Float64}}()
    atrsols = Vector{Array{Float64}}()
    drugsols = Vector{Array{Float64}}()
    int_ts = collect(0:12:tmax)
    for (i, t) in enumerate(int_ts)
        # atn_pkpd = make_scaled_atn_pkpd_model(fbb_ui[1:36] .- fbb_u0[1:36], part[1:36] .- v0[1:36], L[1:36, 1:36], Ld, m, t)
        atn_pkpd = make_scaled_atn_pkpd_model(fbb_ui[37:72] .- fbb_u0[37:72], part[37:72] .- v0[37:72], L[37:72, 37:72], Ld[37:72, 37:72], m, t)

        
        # amyloid_production = 1. / 12
        # tau_transport = 0.2 / 12
        # tau_production = 0.06 /12
        # coupling = 4.5
        # atrophy = 0.1 / 12
        drug_concentration = 400.
        drug_transport = 1.5 / 12
        drug_effect = 0.1 / 12
        drug_clearance = 5. / 12
        _sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36); mean_fbb_init], 
                        (0.0, 360.0), [amyloid_production, tau_transport, tau_production, 
                                                coupling, atrophy, 
                                                drug_transport, drug_effect, 
                                                drug_concentration, drug_clearance]; 
                                                saveat=ts, tol=1e-12)
        push!(sols, _sol)
        push!(absols, Array(_sol[1:36,:]))
        push!(tausols, Array(_sol[37:72,:]))
        push!(atrsols, Array(_sol[73:108,:]))
        push!(drugsols, Array(_sol[109:144,:]))    
    end


# --------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------


    tau_end = [mean(t[rois, end]) for t in tausols]
    atr_end = [mean(t[rois, end]) for t in atrsols]
    ax = Axis(f[i,1], xlabel="t0 / months", ylabel="Biomarker \n change", title="CL = $(Int(round(pos_centiloids,digits=0)))", titlesize=25, titlefont=:regular,
    xlabelsize=20, ylabelsize=20,xticks=collect(0:60:tmax), yticklabelsize=20, xticklabelsize=20,)
    ylims!(ax, 0, 0.075)
    xlims!(ax, 0, 370)
    _ts = collect(12:12:tmax)
    cls = LinRange(0.4, 1.0, length(_ts))
    
    atr_diffs = [atr_end[i + 2] - atr_end[i + 1] for i in 0:length(_ts)-1]
    tau_diffs = [tau_end[i + 2] - tau_end[i + 1] for i in 0:length(_ts)-1]
    vlines!(ax, sol.t[coloc_t[1][2]], color=(:grey, 0.9), linewidth=2.5, label="CT = 0.5")
    # vlines!(tau_t, color=(:grey, 0.9), linestyle=:dash, linewidth=2.5, label="CT = 0.79")
    
    scatter!(ax, _ts, atr_diffs, label="Neurodegeneration", color=[get(atrcmap, c) for c in cls], marker=:utriangle, markersize=15)
    scatter!(ax, _ts, tau_diffs, label="Tau", color=[get(taucmap, c) for c in cls], markersize=15)
    if i < 5
        hideydecorations!(ax, grid=false, ticks=false, ticklabels=false)
        hidexdecorations!(ax, grid=false, ticks=false)
    else
        axislegend(ax, unique=true, position=:rt,  framevisible=false, labelsize=20, patchsize=(20,20))
    end
    end
    f
end
save(projectdir("output/plots/pkpd/coloc-pkpd-cl.png"), f)