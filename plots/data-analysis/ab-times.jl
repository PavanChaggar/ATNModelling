using ATNModelling.DataUtils: baseline_difference, sigmoid
using ATNModelling.ConnectomeUtils: get_parcellation, get_cortex, get_dkt_names
using ADNIDatasets: ADNIDataset
using CairoMakie
using Polynomials: Polynomial, coeffs, roots
using CSV, DataFrames
using DelimitedFiles: readdlm
using ColorSchemes
using DrWatson: projectdir, datadir

#-----------------------------------------------------------------------
# Ab integration
#-----------------------------------------------------------------------
ab_coeffs = readdlm(projectdir("output/analysis-derivatives/ab-derivatives/ab-polynomial-coeffs.csv"))

f = Polynomial(vec(ab_coeffs))

ab = CSV.read(projectdir("output/analysis-derivatives/ab-derivatives/ab-vector-field.csv"), DataFrame);
ab_bin = CSV.read(projectdir("output/analysis-derivatives/ab-derivatives/ab-binned-vector-field.csv"), DataFrame);

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
save(projectdir("output/plots/population-analysis/ab-integrated.pdf"), fig)
#-----------------------------------------------------------------------
# Regional Values
#-----------------------------------------------------------------------

data_path = datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv");

data_df = CSV.read(data_path, DataFrame);
fbp_data = filter(x -> x.TRACER == "FBP", data_df);
dropmissing!(fbp_data, :AMYLOID_STATUS_COMPOSITE_REF)
abpos_df = filter(x -> x["AMYLOID_STATUS_COMPOSITE_REF"] ∈ [0, 1], fbp_data)
dktnames = get_parcellation() |> get_cortex |> get_dkt_names;

data = ADNIDataset(abpos_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF", qc=false)

t_df = CSV.read(projectdir("output/analysis-derivatives/ab-derivatives/ab-times.csv"), DataFrame)
dfparams = CSV.read(projectdir("output/analysis-derivatives/ab-derivatives/ab-params.csv"), DataFrame)

ts_idx = t_df.t_idx
ts = t_df.ab_time
begin
    CairoMakie.activate!()
    cols = Makie.wong_colors()

    node = 29
    roi_df = baseline_difference(data[ts_idx], node)
    roi_vals = roi_df.ab_suvr

    fig = Figure(size=(700, 500), fontsize=25)
    ax = Axis(fig[1,1], 
              xlabel="t / years", xlabelsize=25, xticks=-20:20:80,
              ylabel="SUVR", ylabelsize=25, 
              yticks=0.5:0.25:1.25, yticksize=10)
    CairoMakie.xlims!(ax, 0, 85)
    CairoMakie.ylims!(ax, 0.5, 1.45)
    CairoMakie.scatter!(ts, roi_vals, color=(cols[5], 0.5))
    _params = [dfparams.diff[node], dfparams.alpha[node], dfparams.t50[node], dfparams.u0[node]]
    CairoMakie.lines!(0:1:100, sigmoid(0:1:100, _params), color=cols[1], linewidth=5)
    CairoMakie.hlines!(ax, dfparams.u0[node], color=cols[2], linewidth=2.5)
    CairoMakie.hlines!(ax, dfparams.ui[node], color=cols[2], linewidth=2.5)

    fig
end
save(projectdir("output/plots/population-analysis/ab-inferior-temporal.pdf"), f)

using GLMakie; GLMakie.activate!()
using ColorSchemes, Colors
right_cortical_nodes = filter(x -> x.Hemisphere == "right", get_parcellation() |> get_cortex)
cmap = ColorSchemes.viridis;
# cmap = ColorScheme(sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9));
using Connectomes:get_node_id, plot_roi!
nodes = get_node_id.(right_cortical_nodes)
ui = dfparams.ui
d = (ui .- minimum(ui)) ./ (maximum(ui) .- minimum(ui))
begin
    f = Figure(size = (750, 400))
    ax = Axis3(f[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, d, cmap)
    ax = Axis3(f[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))

    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(nodes, d, cmap)
    Colorbar(f[2, 1:2], limits = (0.8, maximum(ui)), colormap = cmap,
    vertical = false, label = "SUVR", labelsize=25, flipaxis=false,
    ticksize=18, ticklabelsize=20, ticks=0.8:0.1:1.4, labelpadding=3)
end
save(projectdir("output/plots/population-analysis/ab-carrying-capacities.jpeg"), f)