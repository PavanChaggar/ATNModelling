using ATNModelling.SimulationUtils: load_ab_params, load_tau_params,
                                    make_atn_model, make_prob, simulate, resimulate,
                                    generate_data
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names, get_distance_laplacian, get_braak_regions
using ATNModelling.InferenceModels: atn_inference, fit_model
using ATNModelling.DataUtils: baseline_difference, sigmoid
using Connectomes: laplacian_matrix, get_label, get_hemisphere, get_node_id,
                   plot_roi!
using FileIO
using DrWatson: projectdir, datadir
using CSV, DataFrames, ADNIDatasets
using Serialization
using LsqFit, Colors
using StatisticalMeasures, Statistics
# --------------------------------------------------------------------------------
# Simulation set-up
# --------------------------------------------------------------------------------
u0, ui = load_ab_params(tracer="FBB")
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2)
L = laplacian_matrix(c)

α_a, ρ_t, α_t, β, η = 0.75, 0.02, 0.5, 3.21, 0.05
params = [α_a, ρ_t, α_t, β, η]

ab_inits = copy(u0)
tau_inits = copy(v0)
atr_inits = zeros(72)

ab_inits .+= 0.25 .* (ui .- u0)

tau_seed_regions = ["entorhinal" ]#,"Left-Amygdala", "Right-Amygdala", "Left-Hippocampus", "Right-Hippocampus"]
tau_seed_idx = findall(x -> get_label(x) ∈ tau_seed_regions, c.parc)
tau_inits[tau_seed_idx] .+= 0.25 .* ((part[tau_seed_idx] .+ β .* (ui[tau_seed_idx] - u0[tau_seed_idx])) .- v0[tau_seed_idx])

_vi = part .+ (β .* (ui .- u0))
inits = [ab_inits; tau_inits; atr_inits]

tspan = (0.0,30.0)

func = make_atn_model(u0, ui, v0, part, L)

ts = LinRange(0.0, 30, 4)
sol = simulate(func, inits, tspan, params; saveat=0.1)
sol_ts = simulate(func, inits, tspan, params; saveat=ts)

using CairoMakie; CairoMakie.activate!()

plot(sol, idxs=1:72)
plot(sol, idxs=73:144)
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
using GLMakie; GLMakie.activate!()
using Colors, ColorSchemes

_vi = part .+ (3.21 .* (ui .- u0))
cortex = get_parcellation() |> get_cortex
right_cortical_nodes = filter(x -> get_hemisphere(x) == "right", cortex)
nodes = get_node_id.(right_cortical_nodes)

ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
tau_c = ColorScheme(sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5))
tau_c = sequential_palette(250, s = 1, c = 0.9, w =0.25, b = 0.25)
atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);

# begin
#     GLMakie.activate!()
#     f = Figure(size=(1350, 1200))
#     cols = Makie.wong_colors()

#     gt  = f[1,1] = GridLayout()
#     gb  = f[2, 1] = GridLayout()
#     gt1 = gt[1, 1] = GridLayout()
#     gt2 = gt[1, 2] = GridLayout()
#     gt3 = gt[1, 3] = GridLayout()

#     g1 = gb[1, 1] = GridLayout()
#     g2 = gb[2, 1] = GridLayout()
#     g3 = gb[3, 1] = GridLayout()
    
#     # AB carrying capacities
#     cmap = ColorSchemes.viridis;
#     d = (ui .- minimum(ui)) ./ (maximum(ui) .- minimum(ui))
#     ax = Axis3(gt1[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
#     hidedecorations!(ax)
#     hidespines!(ax)
#     plot_roi!(nodes, d[1:36], cmap)
#     ax = Axis3(gt1[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
#     hidedecorations!(ax)
#     hidespines!(ax)
#     plot_roi!(nodes, d[1:36], cmap)

#     Colorbar(gt1[2, 1:2], limits = (minimum(ui), maximum(ui)), colormap = cmap,
#     vertical = false, label = "Aβ SUVR", labelsize=25, flipaxis=false,
#     ticklabelsize=25, ticks=0.8:0.1:1.4)

#     d = (vi .- minimum(vi)) ./ (maximum(vi) .- minimum(vi))
#     ax = Axis3(gt2[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
#     hidedecorations!(ax)
#     hidespines!(ax)
#     plot_roi!(nodes, d[1:36], cmap)
#     ax = Axis3(gt2[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
#     hidedecorations!(ax)
#     hidespines!(ax)
#     plot_roi!(nodes, d[1:36], cmap)

#     Colorbar(gt2[2, 1:2], limits = (minimum(vi), maximum(vi)), colormap = cmap,
#     vertical = false, label = "tau SUVR", labelsize=25, flipaxis=false,
#     ticklabelsize=25, ticks=2.2:0.4:3.8)

#     # AB vs tau correlation 
#     cmap = Makie.wong_colors()
#     v0, vi, part = load_tau_params(tracer="FTP")
#     bf_v0, bf_vi, bf_part = load_tau_params(tracer="RO")
#     fbb_u0, fbb_ui = load_ab_params(tracer="FBB")
#     fbp_u0, fbp_ui = load_ab_params(tracer="FBP")
#     fmm_u0, fmm_ui = load_ab_params(tracer="FMM")
    
#     xs = collect(0:0.1:1.8)
#     linearmodel(x, p) = p[1] .* x
#     fbb_fitted_model = curve_fit(linearmodel, fbb_ui .- fbb_u0, vi .- part, [1.0]);
#     println(fbb_fitted_model.param)
#     println(rsquared(linearmodel(fbb_ui .- fbb_u0, fbb_fitted_model.param), vi .- part))
#     fbp_fitted_model = curve_fit(linearmodel, fbp_ui .- fbp_u0, vi .- part, [1.0])
#     println(fbp_fitted_model.param)
#     println(rsquared(linearmodel(fbp_ui .- fbp_u0, fbp_fitted_model.param), vi .- part))
#     bf_fitted_model = curve_fit(linearmodel, fmm_ui .- fmm_u0, bf_vi .- bf_part, [1.0])
#     println(fbp_fitted_model.param)
#     println(rsquared(linearmodel(fmm_ui .- fmm_u0, bf_fitted_model.param), bf_vi .- bf_part))
#     ax = Axis(gt3[1,1], xlabel="b.c. Aβ SUVR", ylabel="b.c. Tau SUVR", xticks=0:0.5:1.5, xlabelsize=25, 
#                 ylabelsize=25, xticklabelsize=25, yticklabelsize=25)
#     xlims!(ax, 0, 1.5)
#     ylims!(ax, 0, 4)
#     scatter!(fbb_ui .- fbb_u0, vi .- part, color=alphacolor(cmap[1], 0.5), markersize=15, label="FBB")
#     lines!(xs, linearmodel(xs, fbb_fitted_model.param), color=alphacolor(cmap[1], 0.5), linewidth=5)
#     scatter!(fbp_ui .- fbp_u0, vi .- part, color=alphacolor(cmap[2], 0.5), markersize=15, label="FBP")
#     lines!(xs, linearmodel(xs, fbp_fitted_model.param), color=alphacolor(cmap[2], 0.5), linewidth=5)
#     scatter!(fmm_ui .- fmm_u0, bf_vi .- bf_part, color=alphacolor(cmap[3], 0.5), markersize=15, label="FMM")
#     lines!(xs, linearmodel(xs, bf_fitted_model.param), color=alphacolor(cmap[3], 0.5), linewidth=5)
#     axislegend(ax, position=:rb, fontsize=25)
#     ax.alignmode = Mixed(right = 0)

#     # ATN trajectories
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
#     cb = Colorbar(g1[1:2, 8], limits = (minimum(u0), 1.4), colormap = abcmap,
#             vertical = true, labelsize=20, flipaxis=true, ticks=collect(0.6:0.2:1.4),
#             ticksize=18, ticklabelsize=20, labelpadding=10)
#     cb.alignmode = Mixed(right = 0 )
#     cb = Colorbar(g2[1:2, 8], limits = (minimum(v0)-0.1, 3.5), colormap = taucmap,
#             vertical = true, labelsize=20, flipaxis=true, ticks=collect(1.0:0.5:4.0),
#             ticksize=18, ticklabelsize=20, labelpadding=10)
#     cb.alignmode = Mixed(right = 0 )
#     cb = Colorbar(g3[1:2, 8], limits = (0, 1), colormap = atrcmap,
#             vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.2:1.0),
#             ticksize=18, ticklabelsize=20, labelpadding=10)
#     cb.alignmode = Mixed(right = 0 )


#     ts = LinRange(0.0, 24, 5)

#     ax = GLMakie.Axis(g1[1:2,1:2],
#             xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
#             xticklabelsize = 25, xticks = ts, xticksize=10,
#             xlabel="Time / years", xlabelsize = 25,
#             yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
#             yticklabelsize = 25, yticksize=10,
#             ylabel="SUVR", ylabelsize = 25, yticks=collect(0.6:0.2:1.4)
#     )
#     hidexdecorations!(ax, ticks=false, grid=false)
#     GLMakie.ylims!(ax, minimum(u0), 1.4)
#     GLMakie.xlims!(ax, 0, 24)
#     # vlines!(ax, sol.t[coloc_t[2]], color=:black, linewidth=2.0, linestyle=:dash)
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
#     GLMakie.ylims!(ax, minimum(v0)-0.1, 3.5)
#     GLMakie.xlims!(ax, 0, 24)
#     # vlines!(ax, sol.t[coloc_t[2]], color=:black, linewidth=2.0, linestyle=:dash)
#     for i in 1:36
#         lines!(sol.t, sol[i + 72, :], linewidth=2.0, color=alphacolor(tau_col, 0.5))
#     end

#     ax = GLMakie.Axis(g3[1:2,1:2],
#             xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
#             xticklabelsize = 25, xticks = ts, xticksize=10,
#             xlabel="Time / years", xlabelsize = 25,
#             yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
#             yticklabelsize = 25, yticksize=10,
#             ylabel="Atr.", ylabelsize = 25, yticks=collect(0:0.2:1.0)
#         )
#     GLMakie.ylims!(ax, 0, 1.0)
#     GLMakie.xlims!(ax, 0, 24)
#     # vlines!(ax, sol.t[coloc_t[2]], color=:black, linewidth=2.0, linestyle=:dash)
#     for i in 1:36
#         lines!(sol.t, sol[i+144, :], linewidth=2.0, color=alphacolor(atr_col, 0.5))
#     end
#     Label(g1[1:2,0], "A", fontsize=30, tellheight=false)
#     Label(g2[1:2,0], "T", fontsize=30, tellheight=false)
#     Label(g3[1:2,0], "N", fontsize=30, tellheight=false)
    
#     rowsize!(f.layout, 1, 200)
#     rowgap!(f.layout, 1, 50)
#     colgap!(gt, 1, 50)
#     colgap!(gt, 2, 50)

#     colsize!(gt, 1, 400)
#     colsize!(gt, 2, 400)
#     colsize!(gt, 3, 300)


#     Label(gt1[1, 1, TopLeft()], "A", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
#     Label(gt2[1, 1, TopLeft()], "B", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :center, tellheight=false, tellwidth=false)
#     Label(gt3[1, 1, TopLeft()], "C", fontsize = 26, font = :bold, padding = (-40, 0, 0, 0), halign = :center, tellheight=false, tellwidth=false)
#     Label(gb[1, 1, TopLeft()], "D", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    
#     f
# end
# save(projectdir("output/plots/simulation/atn-simulation-all.jpeg"), f)

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
bf_fitted_model = curve_fit(linearmodel, fmm_ui .- fmm_u0, bf_vi .- bf_part, [1.0])
println(fbp_fitted_model.param)
println(rsquared(linearmodel(fmm_ui .- fmm_u0, bf_fitted_model.param), bf_vi .- bf_part))

bs = get_braak_regions()
rois = [findall(x -> get_node_id(x) ∈ b , cortex) for b in bs]

begin
    GLMakie.activate!()
    f = Figure(size=(1350, 1000))
    cols = Makie.wong_colors()
    d = rand(100)
    G  = f[1,1] = GridLayout()
    G1 = G[1:3,1]= GridLayout()
    g1 = G1[1:2,1] = GridLayout()
    g2 = G1[3,1] = GridLayout()
    
    G2 = G[1:3,2] = GridLayout()
    g3 = G2[1,1:3] = GridLayout()
    g4 = G2[2,1:3] = GridLayout()


    cmap = ColorSchemes.viridis;
    d = (ui .- minimum(ui)) ./ (maximum(ui) .- minimum(ui))
    ax = Axis3(g1[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, d[1:36], cmap)
    ax = Axis3(g1[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, d[1:36], cmap)

    Colorbar(g1[2, 1:2], limits = (minimum(ui), maximum(ui)), colormap = cmap,
    vertical = false, label = "Aβ SUVR", labelsize=25, flipaxis=false,
    ticklabelsize=25, ticks=0.8:0.1:1.4)

    d = (vi .- minimum(vi)) ./ (maximum(vi) .- minimum(vi))
    ax = Axis3(g1[3,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, d[1:36], cmap)
    ax = Axis3(g1[3,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, d[1:36], cmap)

    Colorbar(g1[4, 1:2], limits = (minimum(vi), maximum(vi)), colormap = cmap,
    vertical = false, label = "tau SUVR", labelsize=25, flipaxis=false,
    ticklabelsize=25, ticks=2.2:0.4:3.8)

    # Axis(g1[1,1])
    # Axis(g1[2,1])
    # Axis(g1[3,1])
    # Axis(g1[4,1])
    ax = Axis(g2[1,1], xlabel= "Time / years", ylabel="Tau Conentration", yticks=0:0.25:1, xticks=0:10:30, 
              ylabelsize=25, xlabelsize=25, yticklabelsize=25, xticklabelsize=25)
    xlims!(ax, 0, 30)
    ylims!(ax, 0, 1)
    tausol = (Array(sol)[73:144,:] .- v0) ./ (_vi .- v0)
    cmap = reverse(Makie.wong_colors()[1:5])
    labels = ["Braak 1", "Braak 2/3", "Braak 4", "Braak 5", "Braak 6"]
    for (i, (roi, label)) in enumerate(zip(reverse(rois), reverse(labels)))
        lines!(sol.t, vec(mean(tausol[roi,:], dims=1)), color=cmap[i], label=label, linewidth=5)
    end
    axislegend(ax, position=:rb, fontsize=35)
    ax.alignmode = Mixed(left = 0)

    cmap = Makie.wong_colors()
    ax = Axis(g3[1,1], xlabel="b.c. Aβ SUVR", ylabel="b.c. Tau SUVR", xticks=0:0.5:1.5, xlabelsize=25, 
                ylabelsize=25, xticklabelsize=25, yticklabelsize=25)
    xlims!(ax, 0, 1.25)
    ylims!(ax, 0, 3.5)
    scatter!(fbb_ui .- fbb_u0, vi .- part, color=alphacolor(cmap[1], 0.5), markersize=15, label="FBB")
    lines!(xs, linearmodel(xs, fbb_fitted_model.param), color=alphacolor(cmap[1], 0.5), linewidth=5)
    axislegend(ax, position=:rb, fontsize=25)
    ax = Axis(g3[1,2], xlabel="b.c. Aβ SUVR", ylabel="b.c. Tau SUVR", xticks=0:0.5:1.5, xlabelsize=25, 
                ylabelsize=25, xticklabelsize=25, yticklabelsize=25)
    xlims!(ax, 0, 1.25)
    ylims!(ax, 0, 3.5)
    hideydecorations!(ax, grid=false, ticks=false)
    scatter!(fbp_ui .- fbp_u0, vi .- part, color=alphacolor(cmap[2], 0.5), markersize=15, label="FBP")
    lines!(xs, linearmodel(xs, fbp_fitted_model.param), color=alphacolor(cmap[2], 0.5), linewidth=5)
    axislegend(ax, position=:rb, fontsize=25)
    ax = Axis(g3[1,3], xlabel="b.c. Aβ SUVR", ylabel="b.c. Tau SUVR", xticks=0:0.5:1.5, xlabelsize=25, 
                ylabelsize=25, xticklabelsize=25, yticklabelsize=25)
    xlims!(ax, 0, 1.25)
    ylims!(ax, 0, 3.5)
    hideydecorations!(ax, grid=false, ticks=false)
    scatter!(fmm_ui .- fmm_u0, bf_vi .- bf_part, color=alphacolor(cmap[3], 0.5), markersize=15, label="FMM")
    lines!(xs, linearmodel(xs, bf_fitted_model.param), color=alphacolor(cmap[3], 0.5), linewidth=5)
    axislegend(ax, position=:rb, fontsize=25)

    abcmap = ColorScheme(ab_c);
    taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
    atrcmap = ColorScheme(atr_c); #ColorSchemes.Reds;

    ab_col = get(abcmap, 0.75)
    tau_col = get(taucmap, 0.75)
    atr_col = get(atrcmap, 0.75)
    
    for i in 1:4
        absol = (sol_ts[i][1:36] .- minimum(u0[1:36])) ./ (maximum(ui[1:36]) .- minimum(u0[1:36]))
        tausol = (sol_ts[i][73:73+35] .- minimum(v0[1:36])) ./ (maximum(_vi[1:36]) .- minimum(v0[1:36]))
        atrsol = sol_ts[i][145:145+35]
        
        ax = Axis3(g4[1,i+1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, absol, abcmap)
        ax = Axis3(g4[2,i+1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, absol, abcmap)
        
        ax = Axis3(g4[3,i+1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, tausol, taucmap)
        ax = Axis3(g4[4,i+1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, tausol, taucmap)

        ax = Axis3(g4[5,i+1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, atrsol, atrcmap)
        ax = Axis3(g4[6,i+1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
        hidedecorations!(ax)
        hidespines!(ax)
        plot_roi!(nodes, atrsol, atrcmap)
    end
     cb = Colorbar(g4[1:2, 6], limits = (minimum(u0), 1.4), colormap = abcmap,
            vertical = true, labelsize=20, flipaxis=true, ticks=collect(0.6:0.2:1.4),
            ticksize=18, ticklabelsize=20, labelpadding=10)
    cb.alignmode = Mixed(right = 0 )
    cb = Colorbar(g4[3:4, 6], limits = (minimum(v0)-0.1, 3.5), colormap = taucmap,
            vertical = true, labelsize=20, flipaxis=true, ticks=collect(1.0:0.5:4.0),
            ticksize=18, ticklabelsize=20, labelpadding=10)
    cb.alignmode = Mixed(right = 0 )
    cb = Colorbar(g4[5:6, 6], limits = (0, 1), colormap = atrcmap,
            vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.2:1.0),
            ticksize=18, ticklabelsize=20, labelpadding=10)
    cb.alignmode = Mixed(right = 0 )

    ts = LinRange(0.0, 30, 4)

    ax = GLMakie.Axis(g4[1:2,1],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = ts, xticksize=10,
            xlabel="Time / years", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="Aβ SUVR", ylabelsize = 25, yticks=collect(0.6:0.2:1.4)
    )
    hidexdecorations!(ax, ticks=false, grid=false)
    GLMakie.ylims!(ax, minimum(u0), 1.4)
    GLMakie.xlims!(ax, 0, 30)
    # vlines!(ax, sol.t[coloc_t[2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, sol[i, :], linewidth=2.0, color=alphacolor(ab_col, 0.5))
    end

    ax = GLMakie.Axis(g4[3:4,1],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = ts, xticksize=10,
            xlabel="Time / years", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="Tau SUVR", ylabelsize = 25, yticks=collect(1.0:0.5:4.0)
        )
    hidexdecorations!(ax, ticks=false, grid=false)
    GLMakie.ylims!(ax, minimum(v0)-0.1, 3.5)
    GLMakie.xlims!(ax, 0, 30)
    # vlines!(ax, sol.t[coloc_t[2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, sol[i + 72, :], linewidth=2.0, color=alphacolor(tau_col, 0.5))
    end

    ax = GLMakie.Axis(g4[5:6,1],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = ts, xticksize=10,
            xlabel="Time / years", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="Atr.", ylabelsize = 25, yticks=collect(0:0.2:1.0)
        )
    GLMakie.ylims!(ax, 0, 1.0)
    GLMakie.xlims!(ax, 0, 30)
    # vlines!(ax, sol.t[coloc_t[2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, sol[i+144, :], linewidth=2.0, color=alphacolor(atr_col, 0.5))
    end
    
    colsize!(G, 1, 400)
    rowsize!(G1, 1, 200)
    rowsize!(G1, 2, 200)
    rowsize!(G2, 2, 600)
    colsize!(g4, 1, 200)

    colgap!(G, 1, 60)
    rowgap!(G2, 1, 30)

    Label(g1[1, 1, TopLeft()], "A", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    Label(g1[3, 1, TopLeft()], "B", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :center, tellheight=false, tellwidth=false)
    Label(g3[1, 1, TopLeft()], "C", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    Label(g2[1, 1, TopLeft()], "D", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :center, tellheight=false, tellwidth=false)
    Label(g4[1, 1, TopLeft()], "E", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    
    f
end
save(projectdir("output/plots/simulation/atn-simulation-all.jpeg"), f)