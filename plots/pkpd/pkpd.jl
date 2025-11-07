using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, make_scaled_atn_pkpd_model_tau,
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params, conc, make_scaled_atn_pkpd_model
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names, get_distance_laplacian, get_braak_regions,
                                    get_subcortex
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise

using Connectomes: laplacian_matrix, get_label, get_hemisphere, get_node_id, plot_roi!
using ADNIDatasets: ADNIDataset, get_id, get_dates, get_initial_conditions, calc_suvr, get_vol, get_times
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
cortex = filter(x -> get_hemisphere(x) == "right", parc)
subcortex = filter(x -> get_hemisphere(x) == "right", subcortical_parc)

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

# --------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------
_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 
tau_cutoffs = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-2std.csv")) |> vec
ab_threshold = readdlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/ab-thresholds.csv")) |> vec
tau_threshold = readdlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/tau-thresholds.csv")) |> vec

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 && x.NEO_Status == 0, tau_data_df);
_tau_data = ADNIDataset(tau_data_df, dktnames; min_scans=1)
tau_subs = get_id.(_tau_data)

tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
# fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer && x.AMYLOID_STATUS_COMPOSITE_REF == 1 && x.RID ∈ tau_subs && x.CENTILOIDS < 67, _ab_data_df);
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer 
                          && x.AMYLOID_STATUS_COMPOSITE_REF == 1 && x.RID ∈ tau_subs 
                          && x.CENTILOIDS < 80, _ab_data_df);
                        #   && x.CENTILOIDS < 60, _ab_data_df);
mean(fbb_data_df.CENTILOIDS)
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF")

tau_data = filter(x -> get_id(x) ∈ get_id.(fbb_data), _tau_data)

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

vol_init = zeros(36)
tmax = 360
ts = range(0, tmax, tmax * 10)

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

atn_pkpd = make_scaled_atn_pkpd_model(fbb_ui[1:36] .- fbb_u0[1:36], part[1:36] .- v0[1:36], L[1:36, 1:36], Ld, m, 0)

using PrettyTables  
names = ["amyloid_production", "tau_transport", "tau_production", 
                                        "coupling", "atrophy", 
                                        "drug_transport", "drug_effect", 
                                        "drug_concentration", "drug_clearance"]
vals = [amyloid_production, tau_transport, tau_production, 
                                        coupling, atrophy, 
                                        drug_transport, drug_effect, 
                                        drug_concentration, drug_clearance]

df = DataFrame(:Parameter => names, :Value => vals)

formatter = (v, i, j) -> round(v, digits = 3);

pretty_table(df; formatters = ft_printf("%5.3f"), backend=Val(:latex))

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

bs = get_braak_regions()
rbs = [filter(x -> x < 42, b) for b in bs]
b3 = reduce(vcat, rbs[1:3])
rois = findall(x -> get_node_id(x) ∈ b3, cortex)
dktnames[rois]

ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);
abcmap = ColorScheme(ab_c);
taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
atrcmap = ColorScheme(atr_c); #ColorSchemes.Reds;

sols = Vector{ODESolution}()
absols = Vector{Array{Float64}}()
tausols = Vector{Array{Float64}}()
atrsols = Vector{Array{Float64}}()
drugsols = Vector{Array{Float64}}()
int_ts = collect(0:12:tmax)
for (i, t) in enumerate(int_ts)
    atn_pkpd = make_scaled_atn_pkpd_model(fbb_ui[1:36] .- fbb_u0[1:36], part[1:36] .- v0[1:36], L[1:36, 1:36], Ld, m, t)

    
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
begin
    GLMakie.activate!()
    f = Figure(size=(1400, 1400), figure_padding=(20,20,20,20))
    nodes = get_node_id.(cortex)
    subcortical_nodes = get_node_id.(subcortex)
    lcmap = Makie.wong_colors()
    cmap = ColorSchemes.viridis
    scmap = ColorSchemes.Greys
    gg = f[1, 1:2] = GridLayout(alignmode = Mixed(
        left = Makie.Protrusion(0),
        right = Makie.Protrusion(0),
        bottom = Makie.Protrusion(35),
        top = Makie.Protrusion(10)))
    g1 = gg[1, 1]  = GridLayout()
    
    ax1 = Axis(g1[1,1], ylabel="Drug μg / ml ", ytickformat="{:.1f}",
                          xticks=(0:60:360, string.(collect(0:5:30))), yticks=0:200:400,
                          xlabel="Time / Years", yticklabelsize=20,
    xticklabelsize=20, ylabelpadding=10, xlabelsize=20, ylabelsize=20)
    xlims!(ax1, 0, 360)
    ylims!(ax1, 0, 550)
    plot!(sol, idxs=109:144)

    gtr = gg[1,2] = GridLayout()
    ax2 = Axis3(gtr[1,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax2)
    hidespines!(ax2)
    plot_roi!(nodes, drugsols[1][:,end] ./ maximum(drugsols[1][:,end]), cmap)
    plot_roi!(subcortical_nodes, fill(0.5, 5), scmap)
    
    ax3 = Axis3(gtr[1,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax3)
    hidespines!(ax3)
    plot_roi!(nodes, drugsols[1][:,end] ./ maximum(drugsols[1][:,end]), cmap)
    plot_roi!(subcortical_nodes, fill(0.5, 5), scmap)
    
    cb = Colorbar(gtr[2, 1:2], limits = (0, maximum(drugsols[1][:,end])), colormap = cmap, 
                 ticklabelsize=20, vertical=false, flipaxis=false, ticks=0:200:400, label="Drug μg / ml", labelsize=20)
    cb.alignmode = Mixed(left = 10, right = 10 , )

    ax_init_1 = Axis3(gtr[1,3], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax_init_1)
    hidespines!(ax_init_1)
    plot_roi!(nodes, absols[1][:,1] ./ 0.5, abcmap)
    plot_roi!(subcortical_nodes, fill(0.5, 5), scmap)
    ax_init_1 = Axis3(gtr[1,4], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax_init_1)
    hidespines!(ax_init_1)
    plot_roi!(nodes, absols[1][:,1] ./ 0.5, abcmap)
    plot_roi!(subcortical_nodes, fill(0.5, 5), scmap)

    cb = Colorbar(gtr[2, 3:4], limits = (0.0, 0.5), colormap = abcmap, label="Aβ conc.",ticklabelsize=20,
                vertical = false, labelsize=20, flipaxis=false, ticks=collect(0:0.5:1), tellheight=true)
    # cb.alignmode = Mixed(right = 0)
    cb.alignmode = Mixed(left = 10, right = 10 , )
    
    ax_init_2 = Axis3(gtr[1,5], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax_init_2)
    hidespines!(ax_init_2)
    plot_roi!(nodes, tausols[1][:,1] ./ 0.5, taucmap)
    plot_roi!(subcortical_nodes, fill(0.5, 5), scmap)
    ax_init_2 = Axis3(gtr[1,6], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax_init_2)
    hidespines!(ax_init_2)
    plot_roi!(nodes, tausols[1][:,1] ./0.5, taucmap)
    plot_roi!(subcortical_nodes, fill(0.5, 5), scmap)

    cb = Colorbar(gtr[2, 5:6], limits = (0.0, 0.5), colormap = taucmap, label="Tau conc.",ticklabelsize=20,
    vertical = false, labelsize=20, flipaxis=false, ticks=collect(0:0.5:1), tellheight=true)
    # cb.alignmode = Mixed(right = 0)
    cb.alignmode = Mixed(left = 10, right = 10 , )
    colgap!(gtr, 2, 30)
    ## Solutions
    # g = [f[1 + i, 1:2] = GridLayout() for i in 1:3]
    gm = f[2, 1:2] = GridLayout()
    
    ab_col = get(abcmap, 0.75)
    tau_col = get(taucmap, 0.75)
    atr_col = get(atrcmap, 0.75)
    
    absolt = Array(placebo_solts[1:36,end])
    tausolt = Array(placebo_solts[37:72,end])
    atrsolt = Array(placebo_solts[73:108,end])
    
    absol = Array(placebo_sol[1:36,:])
    tausol = Array(placebo_sol[37:72,:])
    atrsol = Array(placebo_sol[73:108,:])

    for (i, (solt, sol, cmap, label)) in enumerate(zip([absolt, tausolt, atrsolt], 
                                                [absol, tausol, atrsol], 
                                                [abcmap, taucmap, atrcmap],
                                                ["Aβ Conc.", "Tau Conc.", "Neuro-\ndegeneration"]))

        ax = GLMakie.Axis(gm[i,1],
                xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.25), xgridwidth = 2,
                xticklabelsize = 20, 
                xticks=(0:60:360, string.(collect(0:5:30))), xticksize=5,
                xlabel="Time / Years", xlabelsize = 20,
                yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.25), ygridwidth = 2,
                yticklabelsize = 20, yticksize=5,
                ylabel=label, ylabelsize = 20, yticks=collect(0:0.5:1)
        )
        if i < 3 
            hidexdecorations!(ax, ticks=false, grid=false)
        end
        # hidexdecorations!(ax, ticks=false, grid=false)
        GLMakie.ylims!(ax, 0., 1.)
        GLMakie.xlims!(ax, 0, 360)
        for i in 1:36
            lines!(placebo_sol.t, sol[i, :], linewidth=2.0, color=alphacolor(get(cmap, 0.75), 0.5))
        end

        ax = Axis3(gm[i,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, solt, cmap)
        plot_roi!(subcortical_nodes, fill(0.5, 5), scmap)
        ax = Axis3(gm[i,3], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, solt, cmap)
        plot_roi!(subcortical_nodes, fill(0.5, 5), scmap)
    
    end

    absolt = Array(solts[1:36,end])
    tausolt = Array(solts[37:72,end])
    atrsolt = Array(solts[73:108,end])
    
    absol = Array(sol[1:36,:])
    tausol = Array(sol[37:72,:])
    atrsol = Array(sol[73:108,:])

    for (i, (solt, sol, cmap)) in enumerate(zip([absolt, tausolt, atrsolt], [absol, tausol, atrsol], [abcmap, taucmap, atrcmap]))

        ax = GLMakie.Axis(gm[i,4],
                xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.25), xgridwidth = 2,
                xticklabelsize = 20, 
                xticks=(0:60:360, string.(collect(0:5:30))), xticksize=5,
                xlabel="Time / Years", xlabelsize = 20,
                yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.25), ygridwidth = 2,
                yticklabelsize = 20, yticksize=5,
                ylabel="Conc.", ylabelsize = 20, yticks=collect(0:0.5:1)
        )
        if i < 3 
            hidexdecorations!(ax, ticks=false, grid=false)
        end
        hideydecorations!(ax, grid=false, ticks=false)
        GLMakie.ylims!(ax, 0., 1.)
        GLMakie.xlims!(ax, 0, 360)
        for j in 1:36
            lines!(placebo_sol.t, sol[j, :], linewidth=2.0, color=alphacolor(get(cmap, 0.75), 0.5))
        end
        
        ax = Axis3(gm[i,5], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, solt, cmap)
    plot_roi!(subcortical_nodes, fill(0.5, 5), scmap)
        ax = Axis3(gm[i,6], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, solt, cmap)
    plot_roi!(subcortical_nodes, fill(0.5, 5), scmap)

        cb = Colorbar(gm[i, 7], limits = (0.0, 1.0), colormap = cmap,
        vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.2:1),
        ticksize=10, ticklabelsize=20, labelpadding=10)
        cb.alignmode = Mixed(right=0,)

    end

    ax_inset = Axis(gm[1, 4],
                width=Relative(2/3),
                height=Relative(2/3),
                halign=0.75, tellheight=false, tellwidth=false,
                valign=0.8, ygridcolor = (:grey, 0.25), xgridcolor = (:grey, 0.25),
                xgridwidth = 2,ygridwidth = 2,
                yticks=0:0.2:0.6, xticks=(0:10:40, string.(collect(0:1:4))))
                ylims!(ax_inset, 0., 0.6)
                xlims!(ax_inset, 0., 40)
    for j in 1:36
        lines!(placebo_sol.t, absol[j, :], linewidth=2.0, color=alphacolor(get(ColorSchemes.Greys, 0.5), 0.5))
    end
    for j in cingulate
        lines!(placebo_sol.t, absol[j, :], linewidth=3.0, color=alphacolor(get(abcmap, 0.75), 0.5))
    end
    translate!.(ax_inset.blockscene, 0, 0, 100)
        # ax.alignmode = Mixed(right=0, left=0)
    
    g2 = f[3,1] = GridLayout()
    ax1 = Axis(g2[1,1], ylabel="Aβ Conc.", 
                ytickformat="{:.1f}", ylabelsize=20, yticklabelsize=20, xticklabelsize=20,
                xlabelsize=20, xticks=([0, 60, 120, 240, 360], ["0", "5", "10", "20", "30"]))
    hidexdecorations!(ax1, grid=false, ticks=false)
    ylims!(ax1, 0.0, 1.05)
    xlims!(ax1, 0.0, tmax)
    ax2 = Axis(g2[2,1], ylabel="Tau Conc.", 
                ytickformat="{:.1f}", xlabel="Time / months", 
                ylabelsize=20, xlabelsize=20, yticklabelsize=20, xticklabelsize=20,
                xticks=([0, 60, 120, 240, 360], ["0", "5", "10", "20", "30"]))
    hidexdecorations!(ax2, grid=false, ticks=false)
    ylims!(ax2, 0.0, 1.05)
    xlims!(ax2, 0.0, tmax)
    ax3 = Axis(g2[3,1], ylabel="Neuro-\ndegeneration", 
                ytickformat="{:.1f}", xlabel="Time / Years", ylabelsize=20, yticklabelsize=20, xticklabelsize=20,
                xlabelsize=20, xticks=([0, 60, 120, 240, 360], ["0", "5", "10", "20", "30"]))
    ylims!(ax3, 0.0, 1.05)
    xlims!(ax3, 0.0, tmax)

    cmap = ColorSchemes.Blues

    labels = ["t0 = 0", "t0 = 5","t0 = 10","t0 = 20","Placebo"]
    solidx = [1, 6, 11, 21, 31]
    for (i, (_absol, _tausol, _atrsol, _drugsol, label)) in enumerate(zip(absols[solidx], tausols[solidx], atrsols[solidx], drugsols[solidx], labels))
        absol = vec(mean(_absol[rois,:], dims=1))
        tausol = vec(mean(_tausol[rois,:], dims=1))
        atrsol = vec(mean(_atrsol[rois,:], dims=1))
        drugsol = vec(mean(_drugsol[rois,:], dims=1)) 
        lines!(ax1, sol.t, vec(absol), linewidth=3, color=get(abcmap, ((i-1)/(4/0.6))+0.4), label=label   )
        lines!(ax2, sol.t, vec(tausol), linewidth=3, color=get(taucmap, ((i-1)/(4/0.6))+0.4))
        lines!(ax3, sol.t, vec(atrsol), linewidth=3, color=get(atrcmap, ((i-1)/(4/0.6))+0.4))
        # lines!(ax4, sol.t, vec(drugsol), linewidth=3, color=get(cmap, ((i-1)/(4/0.6))+0.4) , label = label)
    end
    axislegend(ax1, unique=true, position=:lt,  orientation = :horizontal, framevisible=false, fontsize=5, nbanks=2, patchsize=(30,10), padding=(0,0,0,-5))


    g3 = f[3,2] = GridLayout()
    ax = Axis(g3[1,1], xlabel="Tau", ylabel="Neuro-\ndegeneration", 
    yticks=0:0.25:1, xticks=0:0.25:1, xlabelsize=20,
    ylabelsize=20, yticklabelsize=20, xticklabelsize=20, )
    xlims!(ax, 0., 0.8)
    ylims!(ax, 0, 0.8)
    tau_end = [mean(t[rois, end]) for t in tausols]
    atr_end = [mean(t[rois, end]) for t in atrsols]
    for i in eachindex(tau_end)
        linesegments!([0, tau_end[i]], [atr_end[i], atr_end[i]], linestyle=:dash, color=(:grey, 0.75))
        linesegments!([tau_end[i], tau_end[i]], [0, atr_end[i]], linestyle=:dash, color=(:grey, 0.75))
    end
    cls = LinRange(0.4, 1.0, length(tau_end))
    sc = scatter!(tau_end, atr_end, color=[get(cmap, c) for c in cls], markersize=15)

    ax = Axis(g3[2,1], xlabel="t0 / months", ylabel="Biomarker change", 
    xlabelsize=20, ylabelsize=20,xticks=collect(0:60:tmax), yticklabelsize=20, xticklabelsize=20,)
    ylims!(0, 0.055)
    xlims!(0, 370)
    _ts = collect(12:12:tmax)
    cls = LinRange(0.4, 1.0, length(_ts))
    
    atr_diffs = [atr_end[i + 2] - atr_end[i + 1] for i in 0:length(_ts)-1]
    tau_diffs = [tau_end[i + 2] - tau_end[i + 1] for i in 0:length(_ts)-1]
    vlines!(sol.t[coloc_t[1][2]], color=(:grey, 0.9), linewidth=2.5, label="CT = 0.5")
    # vlines!(tau_t, color=(:grey, 0.9), linestyle=:dash, linewidth=2.5, label="CT = 0.79")
    
    scatter!(ax, _ts, atr_diffs, label="Neurodegeneration", color=[get(atrcmap, c) for c in cls], marker=:utriangle, markersize=15)
    scatter!(ax, _ts, tau_diffs, label="Tau", color=[get(taucmap, c) for c in cls], markersize=15)

    axislegend(ax, unique=true, position=:rt,  framevisible=false, labelsize=20, patchsize=(20,20))

    
    colsize!(gg, 1, 400)
    
    rowsize!(f.layout, 1, 200)
    rowsize!(gm, 1, 100)
    rowsize!(gm, 2, 100)
    rowsize!(gm, 3, 100)
    # [colsize!(_g, 1, 250) for _g in g]
    # [colsize!(_g, 4, 250) for _g in g]
    colsize!(gm, 1, 300)
    colsize!(gm, 4, 300)
    rowgap!(gm, 30)
    # colsize!(gl, 1, 200)
    # colsize!(gr, 1, 200)

    Label(gg[1, 1, TopLeft()], "A", fontsize = 25, font = :bold, padding = (0, 60, 10, 0), halign = :center, tellheight=false, tellwidth=false)
    Label(gtr[1, 3, TopLeft()], "B", fontsize = 25, font = :bold, padding = (0, 0, 10, 0), halign = :center, tellheight=false, tellwidth=false)
    Label(gm[1, 1, TopLeft()],  "C", fontsize = 25, font = :bold, padding = (0, 60, 10, 0), halign = :center, tellheight=false, tellwidth=false)
    # Label(g[1][1, 4, TopLeft()],  "C", fontsize = 25, font = :bold, padding = (0, 60, 10, 0), halign = :center, tellheight=false, tellwidth=false)
    Label(g2[1, 1, TopLeft()], "D", fontsize = 25, font = :bold, padding = (0, 0, 10, 0), halign = :left, tellheight=false, tellwidth=false)
    Label(g3[1, 1, TopLeft()], "E", fontsize = 25, font = :bold, padding = (0, 60, 10, 0), halign = :center, tellheight=false, tellwidth=false)
    Label(g3[2, 1, TopLeft()], "F", fontsize = 25, font = :bold, padding = (0, 60, 10, 0), halign = :center, tellheight=false, tellwidth=false)
    f
end
save(projectdir("output/plots/pkpd/coloc-pkpd-60cl.jpeg"), f)


# t0s = collect(0:24:360)
# # solidx = 1:16
# # int_ts[solidx]
# # t0s = int_ts[solidx]
# int_sols = Vector{ODESolution}()
# for (i, t) in enumerate(t0s)
#     atn_pkpd = make_scaled_atn_pkpd_model(fbb_ui[1:36] .- fbb_u0[1:36], 
#                                           part[1:36] .- v0[1:36], L[1:36, 1:36], Ld, m, t)

#     drug_concentration = 400.
#     drug_transport = 1.5 / 12
#     drug_effect = 0.1 / 12
#     drug_clearance = 5. / 12
#     sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36); mean_fbb_init], 
#                     (0.0, 360.0), [amyloid_production, tau_transport, tau_production, 
#                                             coupling, atrophy, 
#                                             drug_transport, drug_effect, 
#                                             drug_concentration, drug_clearance]; 
#                                             saveat=ts, tol=1e-12)

#     push!(int_sols, sol)   
# end

# conc_to_suvr(c, v0, vi) = (c * (vi .- v0)) + v0
# r_vi = vi[1:36]
# r_v0 = v0[1:36]

# trial_duration = 18
# _inits = [placebo_sol(t) for t in t0s]
# int_inits = [_sol(t) for (_sol,t) in zip(int_sols, t0s)]
# int_outcome = [_sol(t + trial_duration) for (_sol,t) in zip(int_sols, t0s)]
# placebo_outcome = [placebo_sol(t + trial_duration) for t in t0s]

# frontal_rois = findall(x -> contains(get_label(x), "frontal"), cortex)
# tau_rois = collect(37:72)

# int_inits_suvr = [conc_to_suvr.(i[tau_rois], r_v0, r_vi) for i in int_inits]
# int_outcome_suvr = [conc_to_suvr.(i[tau_rois], r_v0, r_vi) for i in int_outcome]
# placebo_outcome_suvr = [conc_to_suvr.(i[tau_rois], r_v0, r_vi) for i in placebo_outcome]

# brois = rois

# int_diff = [mean(io[brois]) - mean(ii[brois]) for (io, ii) in zip(int_outcome_suvr, int_inits_suvr)]
# placebo_diff = [mean(io[brois]) - mean(ii[brois]) for (io, ii) in zip(placebo_outcome_suvr, int_inits_suvr)]

# scatter(placebo_diff .- int_diff)

# begin
#     f = Figure()
#     ax = Axis(f[1,1])
#     for sol in int_sols[[1,6,11,15,16]]
#         plot!(sol, idxs=36+29)
#     end
#     f
# end
# begin
#     CairoMakie.activate!()
#     colors = Makie.wong_colors();
#     # Figure and Axis
#     colors = Makie.wong_colors()
#     fig = Figure()
#     ax = Axis(fig[1,1], xticks = (1:3, ["left", "middle", "right"]),
#             title = "Dodged bars with legend")

#     # Plot
#     barplot!(reduce(vcat, [[int_diff[i],placebo_diff[i]] for i in [1, 6, 11]]))

#     # Legend
#     labels = ["group 1", "group 2", "group 3"]
#     elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
#     title = "Groups"
#     Legend(fig[1,2], elements, labels, title)
#     fig
# end
# fig