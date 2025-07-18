using ATNModelling.SimulationUtils: load_ab_params, load_tau_params,
                                    make_atn_model, make_prob, simulate, resimulate,
                                    generate_data
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names
using ATNModelling.InferenceModels: atn_inference, fit_model
using ATNModelling.DataUtils: baseline_difference, sigmoid
using Connectomes: laplacian_matrix, get_label, get_hemisphere, get_node_id,
                   plot_roi!
using FileIO
using DrWatson: projectdir, datadir
using CSV, DataFrames, ADNIDatasets
using Serialization
using LsqFit, Colors
using StatisticalMeasures
# --------------------------------------------------------------------------------
# Simulation set-up
# --------------------------------------------------------------------------------
u0, ui = load_ab_params(tracer="FBB")
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2)
L = laplacian_matrix(c)

α_a, ρ_t, α_t, β, η = 0.75, 0.02, 0.5, 3.21, 0.1
params = [α_a, ρ_t, α_t, β, η]

ab_inits = copy(u0)
tau_inits = copy(v0)
atr_inits = zeros(72)

ab_inits .+= 0.2 .* (ui .- u0)

tau_seed_regions = ["entorhinal" ]#,"Left-Amygdala", "Right-Amygdala", "Left-Hippocampus", "Right-Hippocampus"]
tau_seed_idx = findall(x -> get_label(x) ∈ tau_seed_regions, c.parc)
tau_inits[tau_seed_idx] .+= 0.2 .* ((part[tau_seed_idx] .+ β .* (ui[tau_seed_idx] - u0[tau_seed_idx])) .- v0[tau_seed_idx])

_vi = part .+ (β .* (ui .- u0))
inits = [ab_inits; tau_inits; atr_inits]

tspan = (0.0,30.0)

func = make_atn_model(u0, ui, v0, part, L)

ts = LinRange(0.0, 24, 5)
sol = simulate(func, inits, tspan, params; saveat=0.1)
sol_ts = simulate(func, inits, tspan, params; saveat=ts)

absol = ( Array(sol[1:72,:]) .- u0 ) ./ ( ui .- u0 ) 
tausol = ( Array(sol[73:144,:]) .- v0 ) ./ ( _vi .- v0 )

tau_seed = findall(x -> x >= 0.1, tausol)
tau_seed_idx = zeros(72, size(sol,2))
tau_seed_idx[tau_seed] .= 1.0
# heatmap(tau_seed_idx)

ab_seed = findall(x -> x >= 0.9, absol)
ab_seed_idx = zeros(72, size(sol,2))
ab_seed_idx[ab_seed] .= 1.0
# heatmap(ab_seed_idx)

ab_tau_coloc = tau_seed_idx .* ab_seed_idx
coloc_t = findfirst(x -> x > 0, sum(ab_tau_coloc, dims=1))
coloc_node = findall(x -> x > 0, ab_tau_coloc[:, coloc_t[2]])
# dktnames[coloc_node][1]
sol.t[coloc_t[2]]
# --------------------------------------------------------------------------------
# AB data
# --------------------------------------------------------------------------------
data_path = datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv");

data_df = CSV.read(data_path, DataFrame);
fbp_data = filter(x -> x.TRACER == "FBB", data_df);
dropmissing!(fbp_data, :AMYLOID_STATUS_COMPOSITE_REF)
abpos_df = filter(x -> x["AMYLOID_STATUS_COMPOSITE_REF"] ∈ [0, 1], fbp_data)
dktnames = get_parcellation() |> get_cortex |> get_dkt_names;

data = ADNIDataset(abpos_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF", qc=false)

t_df = CSV.read(projectdir("output/analysis-derivatives/ab-derivatives/FBB/ab-times.csv"), DataFrame)
dfparams = CSV.read(projectdir("output/analysis-derivatives/ab-derivatives/FBB/ab-params.csv"), DataFrame)

ts_idx = t_df.t_idx
ts = t_df.ab_time
# --------------------------------------------------------------------------------
# Simulation visualisatin
# --------------------------------------------------------------------------------
using GLMakie
using Colors, ColorSchemes

_vi = part .+ (3.21 .* (ui .- u0))
cortex = get_parcellation() |> get_cortex
right_cortical_nodes = filter(x -> get_hemisphere(x) == "right", cortex)
nodes = get_node_id.(right_cortical_nodes)

ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);

begin
    f = Figure(size=(1350, 1200))
    cols = Makie.wong_colors()

    gt  = f[1,1] = GridLayout()
    gb  = f[2, 1] = GridLayout()
    gt1 = gt[1, 1] = GridLayout()
    gt2 = gt[1, 2] = GridLayout()
    gt3 = gt[1, 3] = GridLayout()

    g1 = gb[1, 1] = GridLayout()
    g2 = gb[2, 1] = GridLayout()
    g3 = gb[3, 1] = GridLayout()

    # AB integration 
    node = 29
    roi_df = baseline_difference(data[ts_idx], node)
    roi_vals = roi_df.ab_suvr
    
    ts_idx = t_df.t_idx
    ts = t_df.ab_time
    
    # ax = Axis(gt1[1,1], 
    #             xlabel="t / years", xlabelsize=25, xticks=-20:20:80,
    #             ylabel="SUVR", ylabelsize=25, 
    #             yticks=0.5:0.25:1.25, yticksize=10, 
    #             xticklabelsize=25, yticklabelsize=25)
    # xlims!(ax, 0, 85)
    # ylims!(ax, 0.5, 1.35)
    # scatter!(ts, roi_vals, color=(cols[5], 0.5))
    # _params = [dfparams.diff[node], dfparams.alpha[node], dfparams.t50[node], dfparams.u0[node]]
    # lines!(0:1:100, sigmoid(0:1:100, _params), color=cols[1], linewidth=5)
    # hlines!(ax, dfparams.u0[node], color=cols[2], linewidth=2.5)
    # hlines!(ax, dfparams.ui[node], color=cols[2], linewidth=2.5)
    # ax.alignmode = Mixed(left = 0)
    
    # AB carrying capacities
    cmap = ColorSchemes.viridis;
    d = (ui .- minimum(ui)) ./ (maximum(ui) .- minimum(ui))
    ax = Axis3(gt1[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, d[1:36], cmap)
    ax = Axis3(gt1[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, d[1:36], cmap)

    Colorbar(gt1[2, 1:2], limits = (minimum(ui), maximum(ui)), colormap = cmap,
    vertical = false, label = "Aβ SUVR", labelsize=25, flipaxis=false,
    ticklabelsize=25, ticks=0.8:0.1:1.4)

    d = (vi .- minimum(vi)) ./ (maximum(vi) .- minimum(vi))
    ax = Axis3(gt2[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, d[1:36], cmap)
    ax = Axis3(gt2[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, d[1:36], cmap)

    Colorbar(gt2[2, 1:2], limits = (minimum(vi), maximum(vi)), colormap = cmap,
    vertical = false, label = "tau SUVR", labelsize=25, flipaxis=false,
    ticklabelsize=25, ticks=2.2:0.4:3.8)

    # AB vs tau correlation 
    cmap = Makie.wong_colors()
    v0, vi, part = load_tau_params(tracer="FTP")
    bf_v0, bf_vi, bf_part = load_tau_params(tracer="RO")
    fbb_u0, fbb_ui = load_ab_params(tracer="FBB")
    fbp_u0, fbp_ui = load_ab_params(tracer="FBP")
    fmm_u0, fmm_ui = load_ab_params(tracer="FMM")
    
    xs = collect(0:0.1:1.8)
    linearmodel(x, p) = p[1] .* x
    fbb_fitted_model = curve_fit(linearmodel, fbb_ui .- fbb_u0, vi .- part, [1.0]);
    println(fbb_fitted_model.param)
    println(rsquared(linearmodel(fbb_ui .- fbb_u0, fbb_fitted_model.param), vi .- part))
    fbp_fitted_model = curve_fit(linearmodel, fbp_ui .- fbp_u0, vi .- part, [1.0])
    println(fbp_fitted_model.param)
    println(rsquared(linearmodel(fbp_ui .- fbp_u0, fbp_fitted_model.param), vi .- part))
    ax = Axis(gt3[1,1], xlabel="b.c. Aβ SUVR", ylabel="b.c. Tau SUVR", xticks=0:0.25:0.75, xlabelsize=25, 
                ylabelsize=25, xticklabelsize=25, yticklabelsize=25)
    xlims!(ax, 0, 0.8)
    ylims!(ax, 0, 3)
    scatter!(fbb_ui .- fbb_u0, vi .- part, color=alphacolor(cmap[1], 0.75), markersize=15, label="FBB")
    
    lines!(xs, linearmodel(xs, fbb_fitted_model.param), color=alphacolor(cmap[1], 0.75), linewidth=5)
    scatter!(fbp_ui .- fbp_u0, vi .- part, color=alphacolor(cmap[2], 0.75), markersize=15, label="FBP")
    lines!(xs, linearmodel(xs, fbp_fitted_model.param), color=alphacolor(cmap[2], 0.75), linewidth=5)
    axislegend(ax, position=:lt, fontsize=25)
    ax.alignmode = Mixed(right = 0)

    # ATN trajectories
    abcmap = ColorScheme(ab_c);
    taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
    atrcmap = ColorScheme(atr_c); #ColorSchemes.Reds;

    ab_col = get(abcmap, 0.75)
    tau_col = get(taucmap, 0.75)
    atr_col = get(atrcmap, 0.75)
    
    for i in 1:5
        absol = (sol_ts[i][1:72] .- minimum(u0)) ./ (maximum(ui) .- minimum(u0))
        tausol = (sol_ts[i][73:144] .- minimum(v0)) ./ (maximum(_vi) .- minimum(v0))
        atrsol = sol_ts[i][145:216]
        
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
    cb = Colorbar(g1[1:2, 8], limits = (minimum(u0), 1.4), colormap = abcmap,
            vertical = true, labelsize=20, flipaxis=true, ticks=collect(0.6:0.2:1.4),
            ticksize=18, ticklabelsize=20, labelpadding=10)
    cb.alignmode = Mixed(right = 0 )
    cb = Colorbar(g2[1:2, 8], limits = (minimum(v0)-0.1, 3.5), colormap = taucmap,
            vertical = true, labelsize=20, flipaxis=true, ticks=collect(1.0:0.5:4.0),
            ticksize=18, ticklabelsize=20, labelpadding=10)
    cb.alignmode = Mixed(right = 0 )
    cb = Colorbar(g3[1:2, 8], limits = (0, 1), colormap = atrcmap,
            vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.2:1.0),
            ticksize=18, ticklabelsize=20, labelpadding=10)
    cb.alignmode = Mixed(right = 0 )


    ts = LinRange(0.0, 24, 5)

    ax = GLMakie.Axis(g1[1:2,1:2],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = ts, xticksize=10,
            xlabel="Time / years", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="SUVR", ylabelsize = 25, yticks=collect(0.6:0.2:1.4)
    )
    hidexdecorations!(ax, ticks=false, grid=false)
    GLMakie.ylims!(ax, minimum(u0), 1.4)
    GLMakie.xlims!(ax, 0, 24)
    # vlines!(ax, sol.t[coloc_t[2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, sol[i, :], linewidth=2.0, color=alphacolor(ab_col, 0.5))
    end

    ax = GLMakie.Axis(g2[1:2,1:2],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = ts, xticksize=10,
            xlabel="Time / years", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="SUVR", ylabelsize = 25, yticks=collect(1.0:0.5:4.0)
        )
    hidexdecorations!(ax, ticks=false, grid=false)
    GLMakie.ylims!(ax, minimum(v0)-0.1, 3.5)
    GLMakie.xlims!(ax, 0, 24)
    # vlines!(ax, sol.t[coloc_t[2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, sol[i + 72, :], linewidth=2.0, color=alphacolor(tau_col, 0.5))
    end

    ax = GLMakie.Axis(g3[1:2,1:2],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = ts, xticksize=10,
            xlabel="Time / years", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="Atr.", ylabelsize = 25, yticks=collect(0:0.2:1.0)
        )
    GLMakie.ylims!(ax, 0, 1.0)
    GLMakie.xlims!(ax, 0, 24)
    # vlines!(ax, sol.t[coloc_t[2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, sol[i+144, :], linewidth=2.0, color=alphacolor(atr_col, 0.5))
    end
    Label(g1[1:2,0], "A", fontsize=30, tellheight=false)
    Label(g2[1:2,0], "T", fontsize=30, tellheight=false)
    Label(g3[1:2,0], "N", fontsize=30, tellheight=false)
    
    rowsize!(f.layout, 1, 200)
    rowgap!(f.layout, 1, 50)
    colgap!(gt, 1, 50)
    colgap!(gt, 2, 50)

    colsize!(gt, 1, 400)
    colsize!(gt, 2, 400)
    colsize!(gt, 3, 300)


    Label(gt1[1, 1, TopLeft()], "A", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    Label(gt2[1, 1, TopLeft()], "B", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :center, tellheight=false, tellwidth=false)
    Label(gt3[1, 1, TopLeft()], "C", fontsize = 26, font = :bold, padding = (-40, 0, 0, 0), halign = :center, tellheight=false, tellwidth=false)
    Label(gb[1, 1, TopLeft()], "D", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    
    f
end
save(projectdir("output/plots/simulation/atn-simulation-all.jpeg"), f)

xs = collect(0:0.1:1.8)
linearmodel(x, p) = p[1] .* x
fbb_fitted_model = curve_fit(linearmodel, fbb_ui .- fbb_u0, vi .- part, [1.0]);
println(fbb_fitted_model.param)
println(rsquared(linearmodel(fbb_ui .- fbb_u0, fbb_fitted_model.param), vi .- part))
fbp_fitted_model = curve_fit(linearmodel, fbp_ui .- fbp_u0, vi .- part, [1.0])
println(fbp_fitted_model.param)
println(rsquared(linearmodel(fbp_ui .- fbp_u0, fbp_fitted_model.param), vi .- part))
begin
    f = Figure(size=(600, 500))
    ax = Axis(f[1,1], xlabel="b.c. Aβ SUVR", ylabel="b.c. Tau SUVR", xticks=0:0.25:0.75, xlabelsize=25, 
                ylabelsize=25, xticklabelsize=25, yticklabelsize=25)
    xlims!(ax, 0, 1.8)
    ylims!(ax, 0, 4)
    # scatter!(fbb_ui .- fbb_u0, vi .- part, color=alphacolor(cmap[1], 0.75), markersize=15, label="FBB")

    # lines!(xs, linearmodel(xs, fbb_fitted_model.param), color=alphacolor(cmap[1], 0.75), linewidth=5)
    scatter!(fbp_ui .- fbp_u0, vi .- part, color=alphacolor(cmap[2], 0.75), markersize=15, label="FBP")
    lines!(xs, linearmodel(xs, fbp_fitted_model.param), color=alphacolor(cmap[2], 0.75), linewidth=5)
    axislegend(ax, position=:lt, fontsize=25)
    f
end


using Turing, LinearAlgebra

# @model function abtau(ui, vi)
#     σ ~ InverseGamma(2, 3)

#     β ~ Normal(0, 5)

#     vi ~ MvNormal(β .* ui, σ^2 * I)
# end

# m = abtau(fbp_ui .- fbp_u0, vi .- part)
# m()
# sample(m, NUTS(), 2000)

# m()
# begin
#     f = Figure(size=(1250, 800))

#     g1 = f[1, 1:7] = GridLayout()
#     g2 = f[2, 1:7] = GridLayout()
#     g3 = f[3, 1:7] = GridLayout()

#     abcmap = ColorScheme(ab_c);
#     taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
#     atrcmap = ColorScheme(atr_c); #ColorSchemes.Reds;

#     ab_col = get(abcmap, 0.75)
#     tau_col = get(taucmap, 0.75)
#     atr_col = get(atrcmap, 0.75)
    
#     for i in 1:5
#         absol = (sol_ts[i][1:72] .- minimum(u0)) ./ (maximum(ui) .- minimum(u0))
#         tausol = (sol_ts[i][73:144] .- minimum(v0)) ./ (maximum(_vi) .- minimum(v0))
#         atrsol = sol_ts[i][145:216]
        
#         ax = Axis3(g1[1,i+2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#         hidedecorations!(ax)
#         hidespines!(ax)
#         plot_roi!(nodes, absol, abcmap)
#         ax = Axis3(g1[2,i+2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#         hidedecorations!(ax)
#         hidespines!(ax)
#         plot_roi!(nodes, absol, abcmap)
        
#         ax = Axis3(g2[1,i+2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#         hidedecorations!(ax)
#         hidespines!(ax)
#         plot_roi!(nodes, tausol, taucmap)
#         ax = Axis3(g2[2,i+2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#         hidedecorations!(ax)
#         hidespines!(ax)
#         plot_roi!(nodes, tausol, taucmap)

#         ax = Axis3(g3[1,i+2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#         hidedecorations!(ax)
#         hidespines!(ax)
#         plot_roi!(nodes, atrsol, atrcmap)
#         ax = Axis3(g3[2,i+2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#         hidedecorations!(ax)
#         hidespines!(ax)
#         plot_roi!(nodes, atrsol, atrcmap)
#     end
#     Colorbar(g1[1:2, 8], limits = (minimum(u0), 1.3), colormap = abcmap,
#             vertical = true, labelsize=20, flipaxis=true, ticks=collect(0.5:0.25:1.5),
#             ticksize=18, ticklabelsize=20, labelpadding=10)
#     Colorbar(g2[1:2, 8], limits = (minimum(v0), 3.5), colormap = taucmap,
#             vertical = true, labelsize=20, flipaxis=true, ticks=collect(1.0:0.5:4.0),
#             ticksize=18, ticklabelsize=20, labelpadding=10)
#     Colorbar(g3[1:2, 8], limits = (0, 1), colormap = atrcmap,
#             vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.2:1.0),
#             ticksize=18, ticklabelsize=20, labelpadding=10)

#     ax = GLMakie.Axis(g1[1:2,1:2],
#             xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
#             xticklabelsize = 25, xticks = ts, xticksize=10,
#             xlabel="Time / years", xlabelsize = 25,
#             yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
#             yticklabelsize = 25, yticksize=10,
#             ylabel="SUVR", ylabelsize = 25, yticks=collect(0.5:0.25:1.5)
#     )
#     hidexdecorations!(ax, ticks=false, grid=false)
#     GLMakie.ylims!(ax, 0.5-0.05, 1.3+0.05)
#     GLMakie.xlims!(ax, 0, 20)
#     # vlines!(ax, sol.t[arrival_order[1][2]], color=:black, linewidth=2.0, linestyle=:dash)
#     for i in 1:36
#         lines!(sol.t, sol[i, :], linewidth=2.0, color=alphacolor(ab_col, 0.5))
#     end

#     ax = GLMakie.Axis(g2[1:2,1:2],
#             xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
#             xticklabelsize = 25, xticks = ts, xticksize=10,
#             xlabel="Time / years", xlabelsize = 25,
#             yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
#             yticklabelsize = 25, yticksize=10,
#             ylabel="SUVR", ylabelsize = 25, yticks=collect(1.0:0.5:4.0)
#         )
#     hidexdecorations!(ax, ticks=false, grid=false)
#     GLMakie.ylims!(ax, minimum(v0)-0.15, 3.5+0.15)
#     GLMakie.xlims!(ax, 0, 20)
#     # vlines!(ax, sol.t[arrival_order[1][2]], color=:black, linewidth=2.0, linestyle=:dash)
#     for i in 1:36
#     lines!(sol.t, sol[i + 72, :], linewidth=2.0, color=alphacolor(tau_col, 0.5))
#     end

#     ax = GLMakie.Axis(g3[1:2,1:2],
#             xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
#             xticklabelsize = 25, xticks = ts, xticksize=10,
#             xlabel="Time / years", xlabelsize = 25,
#             yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
#             yticklabelsize = 25, yticksize=10,
#             ylabel="Atr.", ylabelsize = 25, yticks=collect(0:0.2:1.0)
#         )
#     GLMakie.ylims!(ax, 0-0.05, 1.0+0.05)
#     GLMakie.xlims!(ax, 0, 20)
#     # vlines!(ax, sol.t[arrival_order[1][2]], color=:black, linewidth=2.0, linestyle=:dash)
#     for i in 1:36
#     lines!(sol.t, sol[i+144, :], linewidth=2.0, color=alphacolor(atr_col, 0.5))
#     end
#     Label(f[1,0], "A", fontsize=30, tellheight=false)
#     Label(f[2,0], "T", fontsize=30, tellheight=false)
#     Label(f[3,0], "N", fontsize=30, tellheight=false)
#     f
# end
# save(projectdir("output/plots/simulation/atn-simulation.jpeg"), f)
