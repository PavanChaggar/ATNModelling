using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, make_scaled_atn_pkpd_model_tau,
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params, conc, make_scaled_atn_pkpd_model
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names, get_distance_laplacian, get_braak_regions
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise

using Connectomes: laplacian_matrix, get_label, get_hemisphere, get_node_id, plot_roi!
using ADNIDatasets: ADNIDataset, get_id, get_dates, get_initial_conditions, calc_suvr, get_vol, get_times
using DrWatson: projectdir, datadir
using CSV, DataFrames
using CairoMakie, Colors, ColorSchemes, GLMakie
using Statistics, SciMLBase
using LinearAlgebra
using DelimitedFiles
# --------------------------------------------------------------------------------
# Tracer independent data
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params()
parc = get_parcellation() |> get_cortex
cortex = filter(x -> get_hemisphere(x) == "right", parc)

c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
Ld = get_distance_laplacian()

cingulate = findall(x -> contains(get_label(x), "cingulate"), cortex)
m = zeros(36)
m[cingulate] .= 1
m[[35,36]] .= 1

dktnames = get_parcellation() |> get_cortex |> get_dkt_names

using Serialization
pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-fixed-beta-lognormal-1x1000.jls"));
# pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-sepbeta-1x1000.jls"));
meanpst = mean(pst)
mean([meanpst["α_a[$i]", :mean] for i in 1:34])
mean([meanpst["ρ_t[$i]", :mean] for i in 1:34])
mean([meanpst["α_t[$i]", :mean] for i in 1:34])
mean([meanpst["η[$i]", :mean] for i in 1:34])

tau_cutoffs = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-2std.csv")) |> vec

# --------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------
_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 && x.NEO_Status == 0, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=1)
IDS = unique(tau_pos_df.RID)

tracer="FBB"

fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer && x.RID ∈ IDS, _ab_data_df)
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF")

fbb_suvr = calc_suvr.(fbb_data)
normalise!(fbb_suvr, fbb_u0, fbb_ui)
fbb_conc = map(x -> conc.(x, fbb_u0, fbb_ui), fbb_suvr)
fbb_inits = [d[:,1] for d in fbb_conc]
mean_fbb_init = mean(fbb_inits)[1:36]
println("ab concentration = $(mean_fbb_init[29])")

fbb_tau_suvr = calc_suvr.(tau_data)
vi = part .+ (3.2258211441306877 .* (fbb_ui .- fbb_u0))
# vi = part .+ (4.5.* (fbb_ui .- fbb_u0))
normalise!(fbb_tau_suvr, v0, vi)
fbb_tau_conc = map(x -> conc.(x, v0, vi), fbb_tau_suvr)
fbb_tau_inits = [d[:,1] for d in fbb_tau_conc]

_mean_tau_init = mean(fbb_tau_inits)
idx = _mean_tau_init .< conc.(tau_cutoffs, v0, vi)
_mean_tau_init[idx] .= 0
mean_tau_init = maximum.(zip(_mean_tau_init[1:36], _mean_tau_init[37:end]))

println("tau concentration = $(mean_tau_init[29])")
scatter(mean_tau_init)

# vol_init = zeros(36)

# cingulate = findall(x -> contains(get_label(x), "cingulate"), cortex)
# m = zeros(36)
# m[cingulate] .= 1
# m[[35,36]] .= 1

amyloid_production = 0.35 / 12
tau_transport = 0.05 / 12
tau_production = 0.2 / 12
coupling = 3.2258211441306877
atrophy = 0.1 / 12
drug_concentration = 100.
drug_transport = 0.5 / 12
drug_effect = 0.1 / 12
drug_clearance = 0.5 / 12
# amyloid_production = 0.2
# tau_transport = 0.01 
# tau_production = 0.04 
# coupling = 4.5 
# atrophy = 0.05 
# drug_concentration = 10.
# drug_transport = 0.1
# drug_effect = 0.0
# drug_clearance = 0.1

# tau_init = zeros(36)
# tau_init[27] = 0.2

ts = range(0,  360, 600)
# ts = range(0, 180, 480)

atn_pkpd = make_scaled_atn_pkpd_model(fbb_ui[1:36] .- fbb_u0[1:36], part[1:36] .- v0[1:36], L[1:36, 1:36], Ld, m, 0)
# atn_pkpd = make_scaled_atn_pkpd_model_tau(fbb_ui[1:36] .- fbb_u0[1:36], part[1:36] .- v0[1:36], L[1:36, 1:36], Ld, m, 0)

sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                (0.0, 360.0), [amyloid_production, tau_transport, tau_production, 
                                        coupling, atrophy, 
                                        drug_transport, drug_effect, 
                                        drug_concentration, drug_clearance]; 
                                        saveat=ts, tol=1e-9)

solts = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                (0.0, 360.0), [amyloid_production, tau_transport, tau_production, 
                        coupling, atrophy, 
                        drug_transport, drug_effect, 
                        drug_concentration, drug_clearance]; 
                        saveat=ts, tol=1e-9)

placebo_sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                (0.0, 360.0), [amyloid_production, tau_transport, tau_production, 
                                        coupling, atrophy, 
                                        drug_transport, 0.0, 
                                        drug_concentration, drug_clearance]; 
                                        saveat=ts, tol=1e-9)

placebo_solts = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                (0.0, 360.0), [amyloid_production, tau_transport, tau_production, 
                        coupling, atrophy, 
                        drug_transport, 0.0, 
                        drug_concentration, drug_clearance]; 
                        saveat=ts, tol=1e-9)

absol = Array(sol[1:36,:])
tausol = Array(sol[37:72,:])
atrsol = Array(sol[73:108,:])
drugsol = Array(sol[109:end,:]) 

begin
    CairoMakie.activate!()
    f = Figure()
    ax1 = Axis(f[1,1], ylabel="Amyloid Conc", ytickformat="{:.0f}")
    hidexdecorations!(ax1, grid=false)
    ylims!(ax1, 0.0, 1.05)
    ax2 = Axis(f[2,1], ylabel="Tau Conc", ytickformat="{:.0f}", xlabel="Time / Months")
    hidexdecorations!(ax2, grid=false)
    ylims!(ax2, 0.0, 1.05)
    ax3 = Axis(f[3,1], ylabel="Atr", ytickformat="{:.0f}", xlabel="Time / Months")
    ylims!(ax3, 0.0, 1.05)
    hidexdecorations!(ax3, grid=false)
    ax4 = Axis(f[4,1], ylabel="Drug Conc \n μg / ml", ytickformat="{:.0f}", xlabel="Time / Months")
    # ylims!(ax4, 0.0, 1.05)
    for i in 1:36
        lines!(ax1, sol.t, absol[i,:])
        lines!(ax2, sol.t, tausol[i,:])
        lines!(ax3, sol.t, atrsol[i,:])
        lines!(ax4, sol.t, drugsol[i,:])
    end
    f
end

using GLMakie

begin 
    GLMakie.activate!()
    nodes = get_node_id.(cortex)
    cmap = ColorSchemes.viridis
    lcmap = Makie.wong_colors()
    f = Figure(size=(1400, 500), fontsize=20)

    g1 = f[1, 1] = GridLayout()
    g2 = f[1, 2] = GridLayout()

    ax1 = Axis(g1[1,1], ylabel="Drug Conc.", ytickformat="{:.1f}", xlabel="Time / Months", yticklabelsize=25,
    xticklabelsize=25)
    xlims!(ax1, 0, 120)
    # ylims!(ax1, 0, 1)
    for i in 1:36
        lines!(ax1, sol.t, drugsol[i,:] ./ 420 , color=lcmap[1], linewidth=2)
    end 
    x = Observable(0.0)
    vlines!(ax1, x, color=(:red, 0.5), linewidth=5)
   
    p1 = Vector{Mesh}(undef, length(nodes))
    p2 = Vector{Mesh}(undef, length(nodes))

    ax2 = Axis3(g2[1,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax2)
    hidespines!(ax2)
    p1 .= plot_roi!(nodes, drugsol[:,1] ./ 420, cmap)
    ax3 = Axis3(g2[2,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax3)
    hidespines!(ax3)
    p2 .= plot_roi!(nodes, drugsol[:,1] ./ 420, cmap)
    # Colorbar(g1[1:2, 8], limits = (minimum(u0), 1.3), colormap = abcmap,
    #         vertical = true, labelsize=20, flipaxis=true, ticks=collect(0.5:0.25:1.5),
    #         ticksize=18, ticklabelsize=20, labelpadding=10)
    cb = Colorbar(g2[:, 2], limits = (0, 1), colormap = cmap, ticklabelsize=25, vertical=true, flipaxis=true)
    colsize!(f.layout, 1, 750)
    colgap!(f.layout, 0)
    colgap!(g2, 0)
    rowgap!(g2, 0)
    cb.alignmode = Mixed(top = 0, bottom= 0)
    f
end
save(projectdir("output/plots/pkpd/central-pkpd.jpeg"), f)

frames = 10 * 48

record(f, projectdir("output/plots/pkpd/pkpd-video.mp4"), 1:frames; framerate=48) do i
    x[] = sol.t[i]
    for k in 1:36
        p1[k].color[] = get(cmap, drugsol[k,i] ./ 420)
        p2[k].color[] = get(cmap, drugsol[k,i] ./ 420)
    end
end

nodes = get_node_id.(cortex)
cmap = ColorSchemes.viridis;
begin 
    GLMakie.activate!()
    p1 = Vector{Mesh}(undef, length(nodes))
    p2 = Vector{Mesh}(undef, length(nodes))

    p3 = Vector{Mesh}(undef, length(nodes))
    p4 = Vector{Mesh}(undef, length(nodes))

    p5 = Vector{Mesh}(undef, length(nodes))
    p6 = Vector{Mesh}(undef, length(nodes))

    f = Figure(size=(1500, 900))
    ax = Axis3(f[1,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0,0,0,0))
    hidedecorations!(ax)
    hidespines!(ax)
    p1 .= plot_roi!(nodes, absol[:,1], cmap)

    ax = Axis3(f[1,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(0,0,0,0))
    hidedecorations!(ax)
    hidespines!(ax)
    p2 .= plot_roi!(nodes, absol[:,1], cmap)

    c = Colorbar(f[1, 0], limits = (0, 1), colormap = cmap,
        vertical = true, label = "Amyloid Conc.", labelsize=25, flipaxis=false,
        ticksize=18, ticklabelsize=25, labelpadding=3,  ticks=0:0.2:1)

    ax1 = Axis(f[1,3:5],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = LinearTicks(5), xticksize=18,
            xlabel="Time / Months", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticks = LinearTicks(5), yticksize=18,
            ylabel="", ylabelsize = 25
    )
    hidexdecorations!(ax1, grid=false, ticks=false)
    ylims!(ax1, 0, 1)
    hidexdecorations!(ax, ticks=false, grid=false)

    for i in 1:36
        lines!(sol.t, absol[i, :], color=Makie.wong_colors()[1])
    end

    ax = Axis3(f[2,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0,0,0,0))
    hidedecorations!(ax)
    hidespines!(ax)
    p3 .= plot_roi!(nodes, tausol[:,1], cmap)

    ax = Axis3(f[2,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(0,0,0,0))
    hidedecorations!(ax)
    hidespines!(ax)
    p4 .= plot_roi!(nodes, tausol[:,1], cmap)

    c = Colorbar(f[2, 0], limits = (0, 1), colormap = cmap,
        vertical = true, label = "Tau Conc.", labelsize=25, flipaxis=false,
        ticksize=18, ticklabelsize=25, labelpadding=3, ticks=0:0.2:1)

    ax2 = Axis(f[2,3:5],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = LinearTicks(5), xticksize=18,
            xlabel="Time / Months", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticks = LinearTicks(5), yticksize=18,
            ylabel="", ylabelsize = 25
    )
    hidexdecorations!(ax2, grid=false, ticks=false)
    ylims!(ax2, 0, 1)
    hidexdecorations!(ax, ticks=false, grid=false)

    for i in 1:36
        lines!(sol.t, tausol[i, :], color=Makie.wong_colors()[1])
    end

    ax = Axis3(f[3,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0,0,0,0))
    hidedecorations!(ax)
    hidespines!(ax)
    p5 .= plot_roi!(nodes, drugsol[:,1], cmap)

    ax = Axis3(f[3,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(0,0,0,0))
    hidedecorations!(ax)
    hidespines!(ax)
    p6 .= plot_roi!(nodes, drugsol[:,1], cmap)

    c = Colorbar(f[3, 0], limits = (0, 1), colormap = cmap,
        vertical = true, label = "Atr.", labelsize=25, flipaxis=false,
        ticksize=18, ticklabelsize=25, labelpadding=3,  ticks=0:0.2:1)

    ax3 = Axis(f[3,3:5],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = LinearTicks(5), xticksize=18,
            xlabel="Time / Months", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticks = LinearTicks(5), yticksize=18,
            ylabel="", ylabelsize = 25
    )
    
    ylims!(ax3, 0, 1)
    for i in 1:36
        lines!(sol.t, atrsol[i, :], color=Makie.wong_colors()[1])
    end

    x = Observable(0.0)
    vlines!(ax1, x, color=(:red, 0.5), linewidth=5)
    vlines!(ax2, x, color=(:red, 0.5), linewidth=5)
    vlines!(ax3, x, color=(:red, 0.5), linewidth=5)

    f
end

frames = 10 * 48

record(f, projectdir("output/plots/pkpd/pkpd-atn-placebo-video.mp4"), 1:frames; framerate=48) do i
    x[] = ts[i]
    for k in 1:36
        p1[k].color[] = get(cmap, absol[k,i])
        p2[k].color[] = get(cmap, absol[k,i])
        p3[k].color[] = get(cmap, tausol[k,i])
        p4[k].color[] = get(cmap, tausol[k,i])
        p5[k].color[] = get(cmap, atrsol[k,i])
        p6[k].color[] = get(cmap, atrsol[k,i])
    end
end

begin
    GLMakie.activate!()
    f = Figure(size=(1000, 800))
    nodes = get_node_id.(cortex)
    g1 = f[1, 1:2] = GridLayout()
    g2 = f[2, 1:2] = GridLayout()
    g3 = f[3, 1:2] = GridLayout()

    ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
    tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
    atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);

    abcmap = ColorScheme(ab_c);
    taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
    atrcmap = ColorScheme(atr_c); #ColorSchemes.Reds;

    ab_col = get(abcmap, 0.75)
    tau_col = get(taucmap, 0.75)
    atr_col = get(atrcmap, 0.75)
    

    absol = Array(placebo_solts[1:36,end])
    tausol = Array(placebo_solts[37:72,end])
    atrsol = Array(placebo_solts[73:108,end])
    
    ax = Axis3(g1[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, absol, abcmap)
    ax = Axis3(g1[2,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, absol, abcmap)
    
    ax = Axis3(g2[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, tausol, taucmap)
    ax = Axis3(g2[2,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, tausol, taucmap)

    ax = Axis3(g3[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, atrsol, atrcmap)
    ax = Axis3(g3[2,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, atrsol, atrcmap)


    absol = Array(placebo_sol[1:36,:])
    tausol = Array(placebo_sol[37:72,:])
    atrsol = Array(placebo_sol[73:108,:])

    # Colorbar(g1[1:2, 6], limits = (0.0, 1.0), colormap = abcmap,
    #         vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.5:1),
    #         ticksize=10, ticklabelsize=20, labelpadding=10)
    # Colorbar(g2[1:2, 6], limits = (0.0, 1.0), colormap = taucmap,
    #         vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.5:1),
    #         ticksize=10, ticklabelsize=20, labelpadding=10)
    # Colorbar(g3[1:2, 6], limits = (0, 1), colormap = atrcmap,
    #         vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.5:1),
    #         ticksize=10, ticklabelsize=20, labelpadding=10)

    ax = GLMakie.Axis(g1[1:2,1],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = [0, 30, 60], xticksize=10,
            xlabel="Time / years", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="Conc.", ylabelsize = 25, yticks=collect(0:0.5:1.)
    )
    hidexdecorations!(ax, ticks=false, grid=false)
    # hideydecorations!(ax, grid=false, ticks=false)

    GLMakie.ylims!(ax, 0., 1.)
    GLMakie.xlims!(ax, 0, 60)
    # ax.alignmode = Mixed(bottom=0)
    # vlines!(ax, sol.t[arrival_order[1][2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, absol[i, :], linewidth=2.0, color=alphacolor(ab_col, 0.5))
    end

    ax = GLMakie.Axis(g2[1:2,1],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = [0, 30, 60], xticksize=10,
            xlabel="Time / months", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="Conc.", ylabelsize = 25, yticks=collect(0:0.5:1.0)
        )
    hidexdecorations!(ax, ticks=false, grid=false)
    # hideydecorations!(ax, grid=false, ticks=false)

    # ax.alignmode = Mixed(bottom=0)
    GLMakie.ylims!(ax, 0, 1)
    GLMakie.xlims!(ax, 0, 60)
    # vlines!(ax, sol.t[arrival_order[1][2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, tausol[i, :], linewidth=2.0, color=alphacolor(tau_col, 0.5))
    end

    ax = GLMakie.Axis(g3[1:2,1],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = [0, 30, 60], xticksize=10,
            xlabel="Time / months", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="Atr.", ylabelsize = 25, yticks=collect(0:0.5:1.0)
        )
    # ax.alignmode = Mixed(bottom=10)
    # hideydecorations!(ax, grid=false, ticks=false)

    GLMakie.ylims!(ax, 0, 1)
    GLMakie.xlims!(ax, 0, 60)
    # vlines!(ax, sol.t[arrival_order[1][2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, atrsol[i, :], linewidth=2.0, color=alphacolor(atr_col, 0.5))
    end
    #  Label(f[1,0], "A", fontsize=30, tellheight=false)
    # Label(f[2,0], "T", fontsize=30, tellheight=false)
    # Label(f[3,0], "N", fontsize=30, tellheight=false)
    colgap!(g1, 0)
    colgap!(g2, 0)
    colgap!(g3, 0)

    [colsize!(g, 1, 225) for g in [g1, g2, g3]]
    [colsize!(g, 2, 225) for g in [g1, g2, g3]]

    g1 = f[1, 3:5] = GridLayout()
    g2 = f[2, 3:5] = GridLayout()
    g3 = f[3, 3:5] = GridLayout()

    ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
    tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
    atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);

    abcmap = ColorScheme(ab_c);
    taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
    atrcmap = ColorScheme(atr_c); #ColorSchemes.Reds;

    ab_col = get(abcmap, 0.75)
    tau_col = get(taucmap, 0.75)
    atr_col = get(atrcmap, 0.75)
    

    absol = Array(solts[1:36,end])
    tausol = Array(solts[37:72,end])
    atrsol = Array(solts[73:108,end])
    
    ax = Axis3(g1[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, absol, abcmap)
    ax = Axis3(g1[2,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, absol, abcmap)
    
    ax = Axis3(g2[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, tausol, taucmap)
    ax = Axis3(g2[2,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, tausol, taucmap)

    ax = Axis3(g3[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, atrsol, atrcmap)
    ax = Axis3(g3[2,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, atrsol, atrcmap)


    absol = Array(sol[1:36,:])
    tausol = Array(sol[37:72,:])
    atrsol = Array(sol[73:108,:])

    Colorbar(g1[1:2, 3], limits = (0.0, 1.0), colormap = abcmap,
            vertical = true, labelsize=25, flipaxis=true, ticks=collect(0:0.5:1),
            ticksize=10, ticklabelsize=25, labelpadding=10)
    Colorbar(g2[1:2, 3], limits = (0.0, 1.0), colormap = taucmap,
            vertical = true, labelsize=25, flipaxis=true, ticks=collect(0:0.5:1),
            ticksize=10, ticklabelsize=25, labelpadding=10)
    Colorbar(g3[1:2, 3], limits = (0, 1), colormap = atrcmap,
            vertical = true, labelsize=25, flipaxis=true, ticks=collect(0:0.5:1),
            ticksize=10, ticklabelsize=25, labelpadding=10)

    ax = GLMakie.Axis(g1[1:2,1],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = [0, 30, 60], xticksize=10,
            xlabel="Time / months", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="Conc.", ylabelsize = 25, yticks=collect(0:0.5:1.)
    )
    hidexdecorations!(ax, ticks=false, grid=false)
    hideydecorations!(ax, grid=false, ticks=false)
    GLMakie.ylims!(ax, 0., 1.)
    GLMakie.xlims!(ax, 0, 60)
    # ax.alignmode = Mixed(bottom=0)
    # vlines!(ax, sol.t[arrival_order[1][2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, absol[i, :], linewidth=2.0, color=alphacolor(ab_col, 0.5))
    end

    ax = GLMakie.Axis(g2[1:2,1],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = [0, 30, 60], xticksize=10,
            xlabel="Time / months", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="Conc.", ylabelsize = 25, yticks=collect(0:0.5:1.0)
        )
    hidexdecorations!(ax, ticks=false, grid=false)
    hideydecorations!(ax, grid=false, ticks=false)
    # ax.alignmode = Mixed(bottom=0)
    GLMakie.ylims!(ax, 0, 1)
    GLMakie.xlims!(ax, 0, 60)
    # vlines!(ax, sol.t[arrival_order[1][2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, tausol[i, :], linewidth=2.0, color=alphacolor(tau_col, 0.5))
    end

    ax = GLMakie.Axis(g3[1:2,1],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = [0, 30, 60], xticksize=10,
            xlabel="Time / months", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="Atr.", ylabelsize = 25, yticks=collect(0:0.5:1.0)
        )
    # ax.alignmode = Mixed(bottom=10)
    hideydecorations!(ax, grid=false, ticks=false)
    GLMakie.ylims!(ax, 0, 1)
    GLMakie.xlims!(ax, 0, 60)
    # vlines!(ax, sol.t[arrival_order[1][2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, atrsol[i, :], linewidth=2.0, color=alphacolor(atr_col, 0.5))
    end
    #  Label(f[1,0], "A", fontsize=30, tellheight=false)
    # Label(f[2,0], "T", fontsize=30, tellheight=false)
    # Label(f[3,0], "N", fontsize=30, tellheight=false)
    colgap!(g1, 0)
    colgap!(g2, 0)
    colgap!(g3, 0)
    # colsize(f.layout, 1, 450)
    [colsize!(g, 1, 225) for g in [g1, g2, g3]]
    [colsize!(g, 2, 225) for g in [g1, g2, g3]]
    
end
save(projectdir("output/plots/pkpd/atn-pkpd-all.jpeg"), f)

ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);

abcmap = ColorScheme(ab_c);
taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
atrcmap = ColorScheme(atr_c); #ColorSchemes.Reds;

begin
    GLMakie.activate!()
    f = Figure(size=(1100, 900), figure_padding=(20,20,20,20))
    nodes = get_node_id.(cortex)
    lcmap = Makie.wong_colors()
    cmap = ColorSchemes.viridis

    gg = f[1, 1:2] = GridLayout(alignmode = Mixed(
        left = Makie.Protrusion(0),
        right = Makie.Protrusion(0),
        bottom = Makie.Protrusion(35),
        top = Makie.Protrusion(10)))
    g1 = gg[1, 1]  = GridLayout()
    ax1 = Axis(g1[1,1:2], ylabel="Drug μg / ml ", ytickformat="{:.1f}", xlabel="Time / Months", yticklabelsize=20,
    xticklabelsize=20, ylabelpadding=10, xlabelsize=20, ylabelsize=20, yticks=0:200:400, xticks=0:120:360)
    xlims!(ax1, 0, 360)
    ylims!(ax1, 0, 450)
    for i in 1:36
        lines!(ax1, sol.t, drugsol[i,:], color=lcmap[1])
    end 
    g2 = gg[1,2] = GridLayout()

    ax2 = Axis3(g2[1,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax2)
    hidespines!(ax2)
    plot_roi!(nodes, drugsol[:,end] ./ maximum(drugsol), cmap)
    ax3 = Axis3(g2[1,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
    hidedecorations!(ax3)
    hidespines!(ax3)
    plot_roi!(nodes, drugsol[:,end] ./ maximum(drugsol), cmap)

    cb = Colorbar(g2[2, 1:2], limits = (0, maximum(drugsol)), colormap = cmap, 
                 ticklabelsize=20, vertical=false, flipaxis=false, ticks=0:200:400, label="Drug μg / ml", labelsize=20)
    cb.alignmode = Mixed(left = 10, right = 10 , )

    g = [f[1 + i, 1:2] = GridLayout() for i in 1:3]

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
                                                ["Aβ Conc.", "Tau Conc.", "Atr."]))

        ax = GLMakie.Axis(g[i][1:2,1],
                xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
                xticklabelsize = 25, xticks = 0:100:300, xticksize=10,
                xlabel="Time / Months", xlabelsize = 25,
                yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
                yticklabelsize = 25, yticksize=10,
                ylabel=label, ylabelsize = 25, yticks=collect(0:0.2:1.)
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

        ax = Axis3(g[i][1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, solt, cmap)
        ax = Axis3(g[i][2,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, solt, cmap)
    
    end

    absolt = Array(solts[1:36,end])
    tausolt = Array(solts[37:72,end])
    atrsolt = Array(solts[73:108,end])
    
    absol = Array(sol[1:36,:])
    tausol = Array(sol[37:72,:])
    atrsol = Array(sol[73:108,:])

    for (i, (solt, sol, cmap)) in enumerate(zip([absolt, tausolt, atrsolt], [absol, tausol, atrsol], [abcmap, taucmap, atrcmap]))

        ax = GLMakie.Axis(g[i][1:2,3],
                xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
                xticklabelsize = 25, xticks = 0:120:360, xticksize=10,
                xlabel="Time / Months", xlabelsize = 25,
                yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
                yticklabelsize = 25, yticksize=10,
                ylabel="Conc.", ylabelsize = 25, yticks=collect(0:0.2:1.)
        )
        if i < 3 
            hidexdecorations!(ax, ticks=false, grid=false)
        end
        hideydecorations!(ax, grid=false, ticks=false)
        GLMakie.ylims!(ax, 0., 1.)
        GLMakie.xlims!(ax, 0, 360)
        for i in 1:36
            lines!(placebo_sol.t, sol[i, :], linewidth=2.0, color=alphacolor(get(cmap, 0.75), 0.5))
        end

        ax = Axis3(g[i][1,4], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, solt, cmap)
        ax = Axis3(g[i][2,4], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, solt, cmap)

        Colorbar(g[i][1:2, 5], limits = (0.0, 1.0), colormap = cmap,
        vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.2:1),
        ticksize=10, ticklabelsize=20, labelpadding=10)
    
    end
    
    colsize!(gg, 1, 600)
    rowsize!(f.layout, 1, 200)
    [colsize!(_g, 1, 300) for _g in g]
    [colsize!(_g, 3, 300) for _g in g]


    Label(g1[1, 1, TopLeft()], "A", fontsize = 30, font = :bold, padding = (10, 40, 50, 0), halign = :center, tellheight=false, tellwidth=false)
    Label(g[1][1, 1, TopLeft()],  "B", fontsize = 30, font = :bold, padding = (0, 40, 50, 0), halign = :center, tellheight=false, tellwidth=false)

    f
end
save(projectdir("output/plots/pkpd/atn-pkpd-all.jpeg"), f)