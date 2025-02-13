using ATNModelling.SimulationUtils: make_scaled_atn_pkpd_model, 
                                    simulate, load_ab_params, load_tau_params, conc
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names, 
                                    get_distance_laplacian

using Connectomes: laplacian_matrix, get_label, get_hemisphere, get_node_id, plot_roi!
using DifferentialEquations
using CairoMakie, GLMakie, ColorSchemes, Colors
using DrWatson
# --------------------------------------------------------------------------------
# Tracer independent data
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params()
u0, ui = load_ab_params(tracer="FBB")
parc = get_parcellation() |> get_cortex
cortex = filter(x -> get_hemisphere(x) == "right", parc)

c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c) 
dktnames = get_parcellation() |> get_cortex |> get_dkt_names
Lh = L[1:36, 1:36]
Ld = get_distance_laplacian()

cingulate = findall(x -> contains(get_label(x), "cingulate"), cortex)
m = zeros(36)
m[cingulate] .= 1
m[[35,36]] .= 1

amyloid_production = 0.2
tau_transport = 0.01
tau_production = 0.04
coupling = 4.5
atrophy = 0.05
drug_concentration = 10.
drug_transport = 0.1
drug_effect = 0.05
drug_clearance = 0.1

tau_init = zeros(36)
tau_init[27] = 0.2
atn_pkpd = make_scaled_atn_pkpd_model(ui[1:36] .- u0[1:36], part[1:36], Lh, Ld, m)

ts = range(0, 60, 600)

sol = simulate(atn_pkpd, [zeros(36) .+ 0.25; tau_init; zeros(72)], 
                (0.0, 60.0), [amyloid_production, tau_transport, tau_production, 
                                        coupling, atrophy, 
                                        drug_transport, drug_effect, 
                                        drug_concentration, drug_clearance]; 
                                        saveat=ts)

solts = simulate(atn_pkpd, [zeros(36) .+ 0.25; tau_init; zeros(72)], 
                (0.0, 60.0), [amyloid_production, tau_transport, tau_production, 
                        coupling, atrophy, 
                        drug_transport, drug_effect, 
                        drug_concentration, drug_clearance]; 
                        saveat=collect(0:30:60))

placebo_sol = simulate(atn_pkpd, [zeros(36) .+ 0.25; tau_init; zeros(72)], 
                (0.0, 60.0), [amyloid_production, tau_transport, tau_production, 
                                        coupling, atrophy, 
                                        drug_transport, 0.0, 
                                        drug_concentration, drug_clearance]; 
                                        saveat=ts)

placebo_solts = simulate(atn_pkpd, [zeros(36) .+ 0.25; tau_init; zeros(72)], 
                (0.0, 60.0), [amyloid_production, tau_transport, tau_production, 
                        coupling, atrophy, 
                        drug_transport, 0.0, 
                        drug_concentration, drug_clearance]; 
                        saveat=collect(0:30:60))

absol = Array(sol[1:36,:])
tausol = Array(sol[37:72,:])
atrsol = Array(sol[73:108,:])
drugsol = Array(sol[109:end,:]) ./ maximum(Array(sol[109:end,:]))

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
    ax4 = Axis(f[4,1], ylabel="Drug Conc", ytickformat="{:.0f}", xlabel="Time / Months")
    ylims!(ax4, 0.0, 1.05)
    for i in 1:36
        lines!(ax1, sol.t, absol[i,:])
        lines!(ax2, sol.t, tausol[i,:])
        lines!(ax3, sol.t, atrsol[i,:])
        lines!(ax4, sol.t, drugsol[i,:])
    end
    f
end

begin 
    GLMakie.activate!()
    nodes = get_node_id.(cortex)
    cmap = ColorSchemes.viridis
    lcmap = Makie.wong_colors()
    f = Figure(size=(1400, 300), fontsize=20)

    g1 = f[1, 1] = GridLayout()
    g2 = f[1, 2] = GridLayout()

    ax1 = Axis(g1[1,1], ylabel="Drug Conc.", ytickformat="{:.1f}", xlabel="Time / Months", yticklabelsize=25,
    xticklabelsize=25)
    xlims!(ax1, 0, 60)
    ylims!(ax1, 0, 1)
    for i in 1:36
        lines!(ax1, sol.t, drugsol[i,:], color=lcmap[1])
    end 
    ax2 = Axis3(g2[1,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax2)
    hidespines!(ax2)
    plot_roi!(nodes, drugsol[:,end], cmap)
    ax3 = Axis3(g2[1,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax3)
    hidespines!(ax3)
    plot_roi!(nodes, drugsol[:,end], cmap)
    # Colorbar(g1[1:2, 8], limits = (minimum(u0), 1.3), colormap = abcmap,
    #         vertical = true, labelsize=20, flipaxis=true, ticks=collect(0.5:0.25:1.5),
    #         ticksize=18, ticklabelsize=20, labelpadding=10)
    cb = Colorbar(g2[:, 3], limits = (0, 1), colormap = cmap, ticklabelsize=25, vertical=true, flipaxis=true)
    colsize!(f.layout, 1, 750)
    colgap!(f.layout, 0)
    colgap!(g2, 0)
    rowgap!(g2, 0)
    cb.alignmode = Mixed(top = 0, bottom= 0)
    f
end
save(projectdir("output/plots/pkpd/central-pkpd.jpeg"), f)

begin
    GLMakie.activate!()
    f = Figure(size=(1400, 800))
    nodes = get_node_id.(cortex)
    g1 = f[1, 1:5] = GridLayout()
    g2 = f[2, 1:5] = GridLayout()
    g3 = f[3, 1:5] = GridLayout()

    ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
    tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
    atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);

    abcmap = ColorScheme(ab_c);
    taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
    atrcmap = ColorScheme(atr_c); #ColorSchemes.Reds;

    ab_col = get(abcmap, 0.75)
    tau_col = get(taucmap, 0.75)
    atr_col = get(atrcmap, 0.75)
    
    for i in 1:3
        absol = Array(placebo_solts[1:36,i])
        tausol = Array(placebo_solts[37:72,i])
        atrsol = Array(placebo_solts[73:108,i])
        
        ax = Axis3(g1[1,i+2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, absol, abcmap)
        ax = Axis3(g1[2,i+2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, absol, abcmap)
        
        ax = Axis3(g2[1,i+2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, tausol, taucmap)
        ax = Axis3(g2[2,i+2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, tausol, taucmap)

        ax = Axis3(g3[1,i+2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, atrsol, atrcmap)
        ax = Axis3(g3[2,i+2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, atrsol, atrcmap)
    end

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

    ax = GLMakie.Axis(g1[1:2,1:2],
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

    ax = GLMakie.Axis(g2[1:2,1:2],
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

    ax = GLMakie.Axis(g3[1:2,1:2],
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


    g1 = f[1, 6:10] = GridLayout()
    g2 = f[2, 6:10] = GridLayout()
    g3 = f[3, 6:10] = GridLayout()

    ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
    tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
    atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);

    abcmap = ColorScheme(ab_c);
    taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
    atrcmap = ColorScheme(atr_c); #ColorSchemes.Reds;

    ab_col = get(abcmap, 0.75)
    tau_col = get(taucmap, 0.75)
    atr_col = get(atrcmap, 0.75)
    
    for i in 1:3
        absol = Array(solts[1:36,i])
        tausol = Array(solts[37:72,i])
        atrsol = Array(solts[73:108,i])
        
        ax = Axis3(g1[1,i+2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, absol, abcmap)
        ax = Axis3(g1[2,i+2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, absol, abcmap)
        
        ax = Axis3(g2[1,i+2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, tausol, taucmap)
        ax = Axis3(g2[2,i+2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, tausol, taucmap)

        ax = Axis3(g3[1,i+2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, atrsol, atrcmap)
        ax = Axis3(g3[2,i+2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, atrsol, atrcmap)
    end

    absol = Array(sol[1:36,:])
    tausol = Array(sol[37:72,:])
    atrsol = Array(sol[73:108,:])

    Colorbar(g1[1:2, 6], limits = (0.0, 1.0), colormap = abcmap,
            vertical = true, labelsize=25, flipaxis=true, ticks=collect(0:0.5:1),
            ticksize=10, ticklabelsize=25, labelpadding=10)
    Colorbar(g2[1:2, 6], limits = (0.0, 1.0), colormap = taucmap,
            vertical = true, labelsize=25, flipaxis=true, ticks=collect(0:0.5:1),
            ticksize=10, ticklabelsize=25, labelpadding=10)
    Colorbar(g3[1:2, 6], limits = (0, 1), colormap = atrcmap,
            vertical = true, labelsize=25, flipaxis=true, ticks=collect(0:0.5:1),
            ticksize=10, ticklabelsize=25, labelpadding=10)

    ax = GLMakie.Axis(g1[1:2,1:2],
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

    ax = GLMakie.Axis(g2[1:2,1:2],
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

    ax = GLMakie.Axis(g3[1:2,1:2],
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
    
end
save(projectdir("output/plots/pkpd/atn-pkpd-all.jpeg"), f)
