using CSV
using DataFrames
using CairoMakie
using Distributions
using Connectomes: get_node_id, plot_roi!
using DelimitedFiles: readdlm
using DrWatson: projectdir, datadir
using ATNModelling.ConnectomeUtils: get_parcellation, get_cortex, get_dkt_names
using ADNIDatasets: ADNIDataset, get_initial_conditions

taudata = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame);

abneg_tau = filter(x -> x.AB_Status == 0, taudata)

cortex = get_parcellation() |> get_cortex
dktnames = get_dkt_names(cortex)

neg_tau_data = ADNIDataset(abneg_tau, dktnames; min_scans=1)
neg_subdata = get_initial_conditions.(neg_tau_data)

pypart = CSV.read(projectdir("output/analysis-derivatives/tau-derivatives/pypart.csv"), DataFrame)

fg(x, μ, σ) = exp.(.-(x .- μ) .^ 2 ./ (2σ^2)) ./ (σ * √(2π))
function plot_density!(μ, Σ, weight; color=:blue, label="")
    d = Normal(μ, sqrt(Σ))
    x = LinRange(quantile(d, .00001),quantile(d, .99999), 200)
    lines!(x, weight .* fg(x, μ, sqrt(Σ)); color = color, label=label)
    band!(x, fill(0, length(x)), weight .* fg(x, μ, sqrt(Σ)); color = (color, 0.1), label=label)
end

node = 27
node_data =   [n[node] for n in neg_subdata]
moments = filter(x -> x.region == node, pypart)
using CairoMakie; CairoMakie.activate!()
cols = Makie.wong_colors();

begin
    f1 = Figure(size=(700, 500), fontsize=25, font = "CMU Serif");
    ax = Axis(f1[1, 1], xlabel="SUVR")
    CairoMakie.xlims!(minimum(node_data) - 0.05, maximum(node_data) + 0.05)
    hist!(vec(node_data), color=(:grey, 0.7), bins=50, normalization=:pdf, label="Data")
    hideydecorations!(ax)
    hidespines!(ax, :t, :r, :l)

    μ = moments.C0_mean[1]
    Σ = moments.C0_cov[1]  
    w = moments.w0
    plot_density!(μ, Σ, w; color=cols[1], label="Healthy")
    vlines!(ax, quantile(Normal(μ, sqrt(Σ)), 0.5), linewidth=3, label=L"p_0", color=cols[1])
    
    μ = moments.C1_mean[1]
    Σ = moments.C1_cov[1]
    w = moments.w1
    plot_density!(μ, Σ, w; color=cols[6], label="Pathological")
    vlines!(ax, quantile(Normal(μ, sqrt(Σ)), 0.99), linewidth=3, label=L"p_\infty", color=cols[6])

    axislegend(; merge = true)
    display(f1)
end
save(projectdir("visualisation/population-analysis/tau-part-gmm-weighted.pdf"), f1)

tau_params = CSV.read(datadir("derivatives/tau-params.csv"), DataFrame)
v0 = tau_params.v0
vi = tau_params.vi

sympart = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/pypart-sym.csv")) |> vec

using GLMakie
using ColorSchemes, Colors
right_cortical_nodes = filter(x -> x.Hemisphere == "right", cortex)
left_cortical_nodes  = filter(x -> x.Hemisphere == "left", cortex)
cmap = ColorSchemes.viridis;

right_nodes = get_node_id.(right_cortical_nodes)
left_nodes = get_node_id.(left_cortical_nodes)

cmap = ColorSchemes.viridis;

d = (sympart .- minimum(v0)) ./ (maximum(vi) .- minimum(v0))

begin
    f = Figure(size = (750, 700))
    
    ax = Axis3(f[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(right_nodes, d[1:36], cmap)
    
    ax = Axis3(f[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(right_nodes, d[1:36], cmap)

    ax = Axis3(f[2,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(left_nodes, d[37:end], cmap)

    ax = Axis3(f[2,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax)
    hidespines!(ax)
    plot_roi!(left_nodes, d[37:end], cmap)

    Colorbar(f[3, 1:2], limits = (minimum(v0), maximum(vi)), colormap = cmap,
    vertical = false, label = "SUVR", labelsize=25, flipaxis=false,
    ticksize=18, ticklabelsize=20, ticks=1:0.5:3.5, labelpadding=3)
    f
end
save("output/plots/population-analysis/part-field.jpeg", f)