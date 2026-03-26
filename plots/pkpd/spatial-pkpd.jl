using ATNModelling.SimulationUtils: load_ab_params, load_tau_params, conc, make_scaled_atn_pkpd_model
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names, get_distance_laplacian
using ATNModelling.DataUtils: normalise!

using Connectomes: laplacian_matrix, get_label, get_hemisphere, get_node_id, plot_roi!
using ADNIDatasets: ADNIDataset, get_id, calc_suvr
using DrWatson: projectdir, datadir
using CSV, DataFrames
using CairoMakie, Colors, ColorSchemes, GLMakie
using Statistics, SciMLBase
using LinearAlgebra
using DelimitedFiles
using Turing
# --------------------------------------------------------------------------------
# Tracer independent data
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params()
parc = get_parcellation() |> get_cortex
cortex = filter(x -> get_hemisphere(x) == "right", parc)

c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
Ld = get_distance_laplacian()
L = laplacian_matrix(c) 

cingulate = findall(x -> contains(get_label(x), "cingulate"), cortex)
m = zeros(36)
m[cingulate] .= 1
m[[35,36]] .= 1

dktnames = get_parcellation() |> get_cortex |> get_dkt_names

using Serialization
# pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-fixed-beta-lognormal-1x1000.jls"));
# pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-sepbeta-1x1000.jls"));
# pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-random-beta-lognormal-1x1000.jls"));
pst = chainscat([deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-random-beta-lognormal-4x1000-$i.jls")) for i in 1:4]...)

meanpst = mean(pst)
mean([meanpst["α_a[$i]", :mean] for i in 1:34])
mean([meanpst["ρ_t[$i]", :mean] for i in 1:34])
mean([meanpst["α_t[$i]", :mean] for i in 1:34])
mean([meanpst["η[$i]", :mean] for i in 1:34])

tau_cutoffs = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-1std.csv")) |> vec

# --------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------
_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 && x.NEO_Status == 0, tau_data_df);
_tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=1)
IDS = unique(tau_pos_df.RID)

tracer="FBB"

fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer 
                          && x.AMYLOID_STATUS_COMPOSITE_REF == 1 && x.RID ∈ IDS 
                          && x.CENTILOIDS < 80, _ab_data_df);
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
println("tau concentration = $(mean_tau_init[29])")
scatter(mean_tau_init)

vol_init = zeros(36)

# cingulate = findall(x -> contains(get_label(x), "cingulate"), cortex)
# m = zeros(36)
# m[cingulate] .= 1
# m[[35,36]] .= 1

amyloid_production = mean([meanpst["α_a[$i]", :mean] for i in 1:18]) / 12
tau_transport = mean([meanpst["ρ_t[$i]", :mean] for i in 1:18]) / 12
tau_production = mean([meanpst["α_t[$i]", :mean] for i in 1:18]) / 12
coupling = meanpst["β_fbb", :mean]
atrophy = mean([meanpst["η[$i]", :mean] for i in 1:18]) / 12

drug_concentration = 400.
drug_transport = 1.5 / 12
drug_effect = 0.1 / 12
drug_clearance = 5. / 12
tmax = 360
ts = range(0,  tmax, tmax*2)
# ts = range(0, 180, 480)

atn_pkpd = make_scaled_atn_pkpd_model(fbb_ui[1:36] .- fbb_u0[1:36], part[1:36] .- v0[1:36], L[1:36, 1:36], Ld, m, 0)
# atn_pkpd = make_scaled_atn_pkpd_model_tau(fbb_ui[1:36] .- fbb_u0[1:36], part[1:36] .- v0[1:36], L[1:36, 1:36], Ld, m, 0)

sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                (0.0, tmax), [amyloid_production, tau_transport, tau_production, 
                                        coupling, atrophy, 
                                        drug_transport, drug_effect, 
                                        drug_concentration, drug_clearance]; 
                                        saveat=ts, tol=1e-9)

solts = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                (0.0, tmax), [amyloid_production, tau_transport, tau_production, 
                        coupling, atrophy, 
                        drug_transport, drug_effect, 
                        drug_concentration, drug_clearance]; 
                        saveat=ts, tol=1e-9)

placebo_sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                (0.0, tmax), [amyloid_production, tau_transport, tau_production, 
                                        coupling, atrophy, 
                                        drug_transport, 0.0, 
                                        drug_concentration, drug_clearance]; 
                                        saveat=ts, tol=1e-9)

placebo_solts = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                (0.0, tmax), [amyloid_production, tau_transport, tau_production, 
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
        plot!(ax1, placebo_sol, idxs=1:36)
        plot!(ax2, placebo_sol, idxs=37:72)
        plot!(ax3, placebo_sol, idxs=73:108)
        plot!(ax4, placebo_sol, idxs=109:144)
    f
end

using GLMakie
ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
tau_c = sequential_palette(240, s = 0.9, c = 0.9, w =0.25, b = 0.5);
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
    ax1 = Axis(g1[1,1:2], ylabel="Drug μg / ml ", ytickformat="{:.1f}",
                          xticks=(0:60:360, string.(collect(0:5:30))), yticks=0:200:400,
                          xlabel="Time / Years", yticklabelsize=20,
    xticklabelsize=20, ylabelpadding=10, xlabelsize=20, ylabelsize=20)
    xlims!(ax1, 0, 360)
    ylims!(ax1, 0, 550)
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
                                                ["Aβ Conc.", "Tau Conc.", "Neurodegeneration"]))

        ax = GLMakie.Axis(g[i][1:2,1],
                xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
                xticklabelsize = 20, 
                xticks=(0:60:360, string.(collect(0:5:30))), xticksize=10,
                xlabel="Time / Years", xlabelsize = 20,
                yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
                yticklabelsize = 20, yticksize=10,
                ylabel=label, ylabelsize = 20, yticks=collect(0:0.2:1.)
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
                xticklabelsize = 20, 
                xticks=(0:60:360, string.(collect(0:5:30))), xticksize=10,
                xlabel="Time / Years", xlabelsize = 20,
                yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
                yticklabelsize = 20, yticksize=10,
                ylabel="Conc.", ylabelsize = 20, yticks=collect(0:0.2:1.)
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

        if i == 1 
        ax_inset = Axis(g[i][1:2, 3],
                        width=Relative(2/3),
                        height=Relative(0.6),
                        halign=5.5/7,
                        valign=0.8, ygridcolor = (:grey, 0.25), xgridcolor = (:grey, 0.25),
                        xgridwidth = 2,ygridwidth = 2,
                        yticks=0:0.2:0.6, xticks=(0:10:40, string.(collect(0:1:4))), backgroundcolor=:white)
                        ylims!(ax_inset, 0., 0.6)
                        xlims!(ax_inset, 0., 40)
            for j in 1:36
                lines!(placebo_sol.t, sol[j, :], linewidth=2.0, color=alphacolor(get(ColorSchemes.Greys, 0.5), 0.5))
            end
            for j in cingulate
                lines!(placebo_sol.t, sol[j, :], linewidth=3.0, color=alphacolor(get(cmap, 0.75), 0.5))
            end
            translate!.(ax_inset.blockscene, 0, 0, 100)
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
save(projectdir("output/plots/pkpd/atn-pkpd-all-360.jpeg"), f)

# begin
#     GLMakie.activate!()
#     f = Figure(size=(1500, 800), figure_padding=(20,20,20,20))
#     nodes = get_node_id.(cortex)
#     lcmap = Makie.wong_colors()
#     cmap = ColorSchemes.viridis

#     gg = f[1, 1:2] = GridLayout(alignmode = Mixed(
#         left = Makie.Protrusion(0),
#         right = Makie.Protrusion(0),
#         bottom = Makie.Protrusion(35),
#         top = Makie.Protrusion(10)))
#     g1 = gg[1, 1]  = GridLayout()
#     ax1 = Axis(g1[1,1:2], ylabel="Drug μg / ml ", ytickformat="{:.1f}",
#                           xticks=(0:60:360, string.(collect(0:5:30))), yticks=0:200:400,
#                           xlabel="Time / Years", yticklabelsize=20,
#     xticklabelsize=20, ylabelpadding=10, xlabelsize=20, ylabelsize=20)
#     xlims!(ax1, 0, 360)
#     ylims!(ax1, 0, 550)
#     for i in 1:36
#         lines!(ax1, sol.t, drugsol[i,:], color=lcmap[1])
#     end 
#     g2 = gg[1,2] = GridLayout()

#     ax2 = Axis3(g2[1,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
#     hidedecorations!(ax2)
#     hidespines!(ax2)
#     plot_roi!(nodes, drugsol[:,end] ./ maximum(drugsol), cmap)
#     ax3 = Axis3(g2[1,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
#     hidedecorations!(ax3)
#     hidespines!(ax3)
#     plot_roi!(nodes, drugsol[:,end] ./ maximum(drugsol), cmap)

#     cb = Colorbar(g2[2, 1:2], limits = (0, maximum(drugsol)), colormap = cmap, 
#                  ticklabelsize=20, vertical=false, flipaxis=false, ticks=0:200:400, label="Drug μg / ml", labelsize=20)
#     cb.alignmode = Mixed(left = 10, right = 10 , )


#     ## Solutions
#     g = [f[1 + i, 1:2] = GridLayout() for i in 1:3]

#     ab_col = get(abcmap, 0.75)
#     tau_col = get(taucmap, 0.75)
#     atr_col = get(atrcmap, 0.75)
    
#     absolt = Array(placebo_solts[1:36,end])
#     tausolt = Array(placebo_solts[37:72,end])
#     atrsolt = Array(placebo_solts[73:108,end])
    
#     absol = Array(placebo_sol[1:36,:])
#     tausol = Array(placebo_sol[37:72,:])
#     atrsol = Array(placebo_sol[73:108,:])

#     for (i, (solt, sol, cmap, label)) in enumerate(zip([absolt, tausolt, atrsolt], 
#                                                 [absol, tausol, atrsol], 
#                                                 [abcmap, taucmap, atrcmap],
#                                                 ["Aβ Conc.", "Tau Conc.", "Neurodegeneration"]))

#         ax = GLMakie.Axis(g[i][1,1],
#                 xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.25), xgridwidth = 2,
#                 xticklabelsize = 20, 
#                 xticks=(0:60:360, string.(collect(0:5:30))), xticksize=10,
#                 xlabel="Time / Years", xlabelsize = 20,
#                 yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.25), ygridwidth = 2,
#                 yticklabelsize = 20, yticksize=10,
#                 ylabel=label, ylabelsize = 20, yticks=collect(0:0.5:1)
#         )
#         if i < 3 
#             hidexdecorations!(ax, ticks=false, grid=false)
#         end
#         # hidexdecorations!(ax, ticks=false, grid=false)
#         GLMakie.ylims!(ax, 0., 1.)
#         GLMakie.xlims!(ax, 0, 360)
#         for i in 1:36
#             lines!(placebo_sol.t, sol[i, :], linewidth=2.0, color=alphacolor(get(cmap, 0.75), 0.5))
#         end

#         ax = Axis3(g[i][1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
#         hidedecorations!(ax)
#         hidespines!(ax)
#         plot_roi!(nodes, solt, cmap)
#         ax = Axis3(g[i][1,3], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
#         hidedecorations!(ax)
#         hidespines!(ax)
#         plot_roi!(nodes, solt, cmap)
    
#     end

#     absolt = Array(solts[1:36,end])
#     tausolt = Array(solts[37:72,end])
#     atrsolt = Array(solts[73:108,end])
    
#     absol = Array(sol[1:36,:])
#     tausol = Array(sol[37:72,:])
#     atrsol = Array(sol[73:108,:])

#     for (i, (solt, sol, cmap)) in enumerate(zip([absolt, tausolt, atrsolt], [absol, tausol, atrsol], [abcmap, taucmap, atrcmap]))

#         ax = GLMakie.Axis(g[i][1,4],
#                 xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.25), xgridwidth = 2,
#                 xticklabelsize = 20, 
#                 xticks=(0:60:360, string.(collect(0:5:30))), xticksize=10,
#                 xlabel="Time / Years", xlabelsize = 20,
#                 yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.25), ygridwidth = 2,
#                 yticklabelsize = 20, yticksize=10,
#                 ylabel="Conc.", ylabelsize = 20, yticks=collect(0:0.5:1)
#         )
#         if i < 3 
#             hidexdecorations!(ax, ticks=false, grid=false)
#         end
#         hideydecorations!(ax, grid=false, ticks=false)
#         GLMakie.ylims!(ax, 0., 1.)
#         GLMakie.xlims!(ax, 0, 360)
#         for j in 1:36
#             lines!(placebo_sol.t, sol[j, :], linewidth=2.0, color=alphacolor(get(cmap, 0.75), 0.5))
#         end
#         if i == 1 
#         ax_inset = Axis(g[i][1, 4],
#                         width=Relative(2/3),
#                         height=Relative(2/3),
#                         halign=0.75,
#                         valign=0.8, ygridcolor = (:grey, 0.25), xgridcolor = (:grey, 0.25),
#                         xgridwidth = 2,ygridwidth = 2,
#                         yticks=0:0.2:0.6, xticks=(0:10:40, string.(collect(0:1:4))))
#                         ylims!(ax_inset, 0., 0.6)
#                         xlims!(ax_inset, 0., 40)
#             for j in 1:36
#                 lines!(placebo_sol.t, sol[j, :], linewidth=2.0, color=alphacolor(get(ColorSchemes.Greys, 0.5), 0.5))
#             end
#             for j in cingulate
#                 lines!(placebo_sol.t, sol[j, :], linewidth=3.0, color=alphacolor(get(cmap, 0.75), 0.5))
#             end
#         end
#         ax = Axis3(g[i][1,5], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
#         hidedecorations!(ax)
#         hidespines!(ax)
#         plot_roi!(nodes, solt, cmap)
#         ax = Axis3(g[i][1,6], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(.0,.0,.0,.0))
#         hidedecorations!(ax)
#         hidespines!(ax)
#         plot_roi!(nodes, solt, cmap)

#         Colorbar(g[i][1, 7], limits = (0.0, 1.0), colormap = cmap,
#         vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.2:1),
#         ticksize=10, ticklabelsize=20, labelpadding=10)
    
#     end
    
#     colsize!(gg, 1, 800)
#     rowsize!(f.layout, 1, 200)
#     [colsize!(_g, 1, 300) for _g in g]
#     [colsize!(_g, 4, 300) for _g in g]


#     Label(g1[1, 1, TopLeft()], "A", fontsize = 30, font = :bold, padding = (10, 40, 50, 0), halign = :center, tellheight=false, tellwidth=false)
#     Label(g[1][1, 1, TopLeft()],  "B", fontsize = 30, font = :bold, padding = (0, 40, 50, 0), halign = :center, tellheight=false, tellwidth=false)

#     f
# end
# save(projectdir("output/plots/pkpd/atn-pkpd-all-360-hor.jpeg"), f)