using ATNModelling.DataUtils: baseline_difference, sigmoid
using ATNModelling.SimulationUtils: load_tau_params, load_ab_params
using ATNModelling.ConnectomeUtils: get_parcellation, get_cortex, get_dkt_names
using ADNIDatasets: ADNIDataset
using CairoMakie
using Polynomials: Polynomial, coeffs, roots
using CSV, DataFrames
using DelimitedFiles: readdlm
using ColorSchemes
using DrWatson: projectdir, datadir
include(projectdir("bf-data.jl"))
#-----------------------------------------------------------------------
# Ab integration
#-----------------------------------------------------------------------
ab_coeffs = readdlm(projectdir("output/analysis-derivatives/bf/ab-derivatives/ab-polynomial-coeffs.csv"))

f = Polynomial(vec(ab_coeffs))

ab = CSV.read(projectdir("output/analysis-derivatives/bf/ab-derivatives/bf-ab-vector-field.csv"), DataFrame);
ab_bin = CSV.read(projectdir("output/analysis-derivatives/bf/ab-derivatives/ab-binned-vector-field.csv"), DataFrame);

begin 
   cols = Makie.wong_colors()
    fig = Figure(size=(700, 500), fontsize=25)
    ax = Axis(fig[1,1], xticks=0.6:0.1:1.8,
              xlabel=L"A\beta \text{ SUVR}", xlabelsize=25, 
              ylabel=L"\Delta A\beta", ylabelsize=25, ylabelrotation=2pi)
    CairoMakie.ylims!(ax, -0.1, 0.1)
    x = LinRange(extrema(ab.ab_suvr)..., 100)
    CairoMakie.scatter!(ab.ab_suvr, ab.ab_diff, color=(cols[5], 0.5), markersize=12)
    CairoMakie.lines!(x, f.(x), color=(cols[3], 1.0), linewidth=5)
    # CairoMakie.lines!(x, f_bin.(x), color=(cols[4], 1.0), linewidth=5)
    CairoMakie.scatter!(roots(f), [0,0], markersize=15, strokewidth=1.0, color=(cols[2]))
    CairoMakie.scatter!(ab_bin.ab_bin, ab_bin.ab_bin_diffs, color=cols[1], strokewidth=1.0, strokecolor=:black, markersize=15)
    fig
end
begin
   cols = Makie.wong_colors()
    fig = Figure(size=(700, 500), fontsize=25)
    ax = Axis(fig[1,1], xticks=0.6:0.1:1.2,
              xlabel=L"A\beta \text{ SUVR}", xlabelsize=25, 
              ylabel=L"\Delta A\beta", ylabelsize=25, ylabelrotation=2pi)
    CairoMakie.ylims!(ax, -0.06, 0.06)
    x = LinRange(extrema(ab.ab_suvr)..., 100)
    CairoMakie.scatter!(ab.ab_suvr, ab.ab_diff, color=(cols[5], 0.5), markersize=12)
    CairoMakie.lines!(x, f.(x), color=(cols[3], 1.0), linewidth=5)
    # CairoMakie.lines!(x, f_bin.(x), color=(cols[4], 1.0), linewidth=5)
    CairoMakie.scatter!(roots(f), [0,0], markersize=15, strokewidth=1.0, color=(cols[2]))
    CairoMakie.scatter!(ab_bin.ab_bin, ab_bin.ab_bin_diffs, color=cols[1], strokewidth=1.0, strokecolor=:black, markersize=15)
    fig
end
save(projectdir("output/plots/population-analysis/ab-integrated-$(tracer).pdf"), fig)
#-----------------------------------------------------------------------
# Regional Values
#-----------------------------------------------------------------------

bf_data_path = datadir("bf-data/bf-data-ab-tau-summary.csv");
bf_data_df = CSV.read(bf_data_path, DataFrame)

bf_ab_data_df = filter(x -> x.ab_status ∈ [0, 1], bf_data_df)
data = BFDataset(bf_ab_data_df, dktnames; min_scans=2, tracer=:ab)

t_df = CSV.read(projectdir("output/analysis-derivatives/bf/ab-derivatives/ab-times.csv"), DataFrame)
dfparams = CSV.read(projectdir("output/analysis-derivatives/bf/ab-derivatives/ab-params.csv"), DataFrame)

ts_idx = t_df.t_idx
ts = t_df.ab_time
for node in 1:36
begin
    CairoMakie.activate!()
    cols = Makie.wong_colors()

    # node = 29
    roi_df = baseline_difference(data[ts_idx], node)
    roi_vals = roi_df.ab_suvr

    fig = Figure(size=(500, 300), fontsize=25)
    ax = Axis(fig[1,1], 
              xlabel="t / years", xlabelsize=25, xticks=-20:20:80,
              ylabel="SUVR", ylabelsize=25, 
              yticks=0.5:0.25:1.25, yticksize=10)
    CairoMakie.xlims!(ax, 0, 85)
    CairoMakie.ylims!(ax, 0.5, 2.45)
    CairoMakie.scatter!(ts, roi_vals, color=(cols[5], 0.5))
    _params = [dfparams.diff[node], dfparams.alpha[node], dfparams.t50[node], dfparams.u0[node]]
    CairoMakie.lines!(0:1:100, sigmoid(0:1:100, _params), color=cols[1], linewidth=5)
    CairoMakie.hlines!(ax, dfparams.u0[node], color=cols[2], linewidth=2.5)
    CairoMakie.hlines!(ax, dfparams.ui[node], color=cols[2], linewidth=2.5)

    display(fig)
end
end
save(projectdir("output/plots/population-analysis/ab-inferior-temporal-$(tracer).pdf"), fig)

using GLMakie; GLMakie.activate!()
using ColorSchemes, Colors
right_cortical_nodes = filter(x -> x.Hemisphere == "right", get_parcellation() |> get_cortex)
left_cortical_nodes = filter(x -> x.Hemisphere == "left", get_parcellation() |> get_cortex)
cmap = ColorSchemes.viridis;
# cmap = ColorScheme(sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9));
using Connectomes:get_node_id, plot_roi!
right_nodes = get_node_id.(right_cortical_nodes)
left_nodes = get_node_id.(left_cortical_nodes)
ui = dfparams.ui
d = (ui .- minimum(ui)) ./ (maximum(ui) .- minimum(ui))

begin
    f = Figure(size = (500, 300))
    ax = Axis3(f[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(right_nodes, d[1:36], cmap)
    ax = Axis3(f[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))

    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(right_nodes, d[1:36], cmap)

    # ax = Axis3(f[2,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    # hidedecorations!(ax)
    # hidespines!(ax)
    # plot_roi!(left_nodes, d[37:end], cmap)
    
    # ax = Axis3(f[2,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    # hidedecorations!(ax)
    # hidespines!(ax)
    # plot_roi!(left_nodes, d[37:end], cmap)

    Colorbar(f[2, 1:2], limits = (minimum(ui), maximum(ui)), colormap = cmap,
    vertical = false, label = "SUVR", labelsize=25, flipaxis=false,
    ticksize=18, ticklabelsize=20, ticks=0.8:0.1:1.4, labelpadding=3)
    
end
save(projectdir("output/plots/population-analysis/ab-carrying-capacities-$(tracer)-2025.jpeg"), f)

#-----------------------------------------------------------------------
# Tau correspondance
#-----------------------------------------------------------------------
v0, vi, part = load_tau_params(tracer="FTP")
bf_v0, bf_vi, bf_part = load_tau_params(tracer="RO")
fbb_u0, fbb_ui = load_ab_params(tracer="FBB")
fbp_u0, fbp_ui = load_ab_params(tracer="FBP")
fmm_u0, fmm_ui = load_ab_params(tracer="FMM")

using LsqFit, Colors

linearmodel(x, p) = p[1] .* x
fbb_fitted_model = curve_fit(linearmodel, fbb_ui .- fbb_u0, vi .- part, [1.0]);
println("params = $(fbb_fitted_model.param)")
fbp_fitted_model = curve_fit(linearmodel, fbp_ui .- fbp_u0, vi .- part, [1.0])
println("params = $(fbp_fitted_model.param)")
fmm_fitted_model = curve_fit(linearmodel, fmm_ui .- fmm_u0, bf_vi .- bf_part, [1.0])
println("params = $(fmm_fitted_model.param)")

begin
    cmap = Makie.wong_colors()
    xs = collect(0:0.1:1.8)
    f = Figure(size=(500, 300), fontsize=25)
    ax = Axis(f[1,1], xlabel="Aβ SUVR", ylabel="Tau SUVR", xticks=0:0.25:0.75)
    xlims!(ax, 0, 0.8)
    ylims!(ax, 0, 4)
    scatter!(fbb_ui .- fbb_u0, vi .- part, color=alphacolor(cmap[1], 0.75), markersize=15, label="FBB")
    lines!(xs, linearmodel(xs, fbb_fitted_model.param), color=alphacolor(cmap[1], 0.75), linewidth=5)
    scatter!(fbp_ui .- fbp_u0, vi .- part, color=alphacolor(cmap[2], 0.75), markersize=15, label="FBP")
    lines!(xs, linearmodel(xs, fbp_fitted_model.param), color=alphacolor(cmap[2], 0.75), linewidth=5)
    # scatter!(fmm_ui .- fmm_u0, bf_vi .- bf_part, color=alphacolor(cmap[3], 0.75), markersize=15, label="FMM")
    # lines!(xs, linearmodel(xs, fmm_fitted_model.param), color=alphacolor(cmap[3], 0.75), linewidth=5)
    axislegend(ax, position=:lt)
    f
end
save(projectdir("output/plots/population-analysis/ab-vs-tau-carrying-capacities.pdf"), f)