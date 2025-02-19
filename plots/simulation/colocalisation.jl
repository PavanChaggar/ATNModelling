using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, simulate,
                                    load_ab_params, load_tau_params, conc,
                                    make_atn_prob_func, atn_output_func,make_scaled_atn_model_hemisphere
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names,
                                    get_braak_regions

using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise

using Connectomes: get_hemisphere, plot_roi!, get_node_id, get_lobe, laplacian_matrix, get_label
using Colors, ColorSchemes, GLMakie
using DifferentialEquations
using CSV, DataFrames, DrWatson
using ADNIDatasets
using Statistics
using Serialization
# --------------------------------------------------------------------------------
# Connectome and Data
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c) 
cortex = get_parcellation() |> get_cortex 
subcortex = filter(x -> get_lobe(x) == "subcortex", get_parcellation())[collect(1:10)]
left_subcortex = filter(x -> get_hemisphere(x) == "left", subcortex)
right_subcortex = filter(x -> get_hemisphere(x) == "right", subcortex)

dktnames = get_dkt_names(cortex)
right_cortex = filter(x -> get_hemisphere(x) == "right", cortex)

# Amyloid data 
_ab_data_df =  CSV.read(datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame) 

tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer && x.AMYLOID_STATUS_COMPOSITE_REF == 1, _ab_data_df);
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF")

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 0 && x.NEO_Status == 0, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=1)

pst = deserialize(projectdir("output/chains/population-scaled-atn/pst-samples-harmonised-dense-1x1000.jls"));
meanpst = mean(pst)
# --------------------------------------------------------------------------------
# Amyloid data
# --------------------------------------------------------------------------------
ab_suvr = calc_suvr.(fbb_data)
normalise!(ab_suvr, fbb_u0, fbb_ui)
ab_conc = map(x -> conc.(x, fbb_u0, fbb_ui), ab_suvr)
ab_inits = [d[:,1] for d in ab_conc]

mean_ab_init = mean(ab_inits)

max_norm(c) =  c ./ maximum(c);

begin
    GLMakie.activate!()
    cmap = ColorSchemes.viridis
    val = mean_ab_init[1:36]
    f = Figure(size = (600, 350), figure_padding = 20, fontsize=25)
    ax = Axis3(f[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex), max_norm(val) , cmap)
    
    ax = Axis3(f[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex), max_norm(val), cmap)
    Colorbar(f[2, 1:2], colormap=cmap, limits=(0,0.5), ticks=0:0.1:0.5,
             ticklabelsize=20, ticksize=10, label="Concentration", vertical = false, flipaxis = false)
    # f
end
save(projectdir("output/plots/colocalisation/mean_ab_aptn.jpeg"), f, px_per_unit=2.0)

tau_suvr = calc_suvr.(tau_data)
normalise!(tau_suvr, v0, vi)
tau_conc = map(x -> conc.(x, v0, vi), tau_suvr)
tau_inits = [d[:,1] for d in tau_conc]

mean_tau_init = mean(tau_inits)
filtered_tau_idx = findall(x -> x < 0.05, mean_tau_init)
mean_tau_init[filtered_tau_idx] .= 0

begin
    GLMakie.activate!()
    cmap = ColorSchemes.viridis
    val = mean_tau_init[1:36]
    f = Figure(size = (600, 350), figure_padding = 20, fontsize=25)
    ax = Axis3(f[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex), max_norm(val) , cmap)
    
    ax = Axis3(f[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex), max_norm(val), cmap)
    Colorbar(f[2, 1:2], colormap=cmap, limits=(0,0.5), ticks=0:0.1:0.5,
             ticklabelsize=20, ticksize=10, label="Concentration", vertical = false, flipaxis = false)
    
end
save(projectdir("output/plots/colocalisation/mean_tau_aptn.jpeg"), f, px_per_unit=2.0)


begin
    GLMakie.activate!()
    cmap = ColorSchemes.viridis
    
    f = Figure(size = (1200, 250), figure_padding = 20, fontsize=25)
    val = mean_ab_init[1:36]
    ax = Axis3(f[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex), max_norm(val) , cmap)
    
    ax = Axis3(f[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex), max_norm(val), cmap)

    val = mean_tau_init[1:36]
    ax = Axis3(f[1,3], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex), max_norm(val) , cmap)
    
    ax = Axis3(f[1,4], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex), max_norm(val), cmap)
    Colorbar(f[1, 0], colormap=cmap, limits=(0,0.5), ticks=0:0.1:0.5,
             ticklabelsize=20, ticksize=10, label="Concentration", vertical = true, flipaxis = false)
    
end
save(projectdir("output/plots/colocalisation/mean_ab_tau_aptn.jpeg"), f, px_per_unit=2.0)


# --------------------------------------------------------------------------------
# Modelling!
# --------------------------------------------------------------------------------
hem_idx = 1:36
atn_model = make_scaled_atn_model_hemisphere((fbb_ui .- fbb_u0)[hem_idx], (part .- v0)[hem_idx], L[hem_idx,hem_idx])

inits = [mean_ab_init[hem_idx]; mean_tau_init[hem_idx]; zeros(36)]
params = meanpst[:Am_a, :mean], meanpst[:Pm_t, :mean], meanpst[:Am_t, :mean], meanpst[:β, :mean], meanpst[:Em, :mean]

sol = simulate(atn_model, inits, (0, 200), params, saveat=0.01)

begin
    CairoMakie.activate!()
    f = Figure(size = (500, 350), figure_padding = 20, fontsize=20)
    ax = Axis(f[1,1], xlabel="t / years", ylabel="Concentration", title="Amyloid Progression",
              yticks=0:0.2:1.0, xticks=0:20:150, yticksize=10)
    CairoMakie.xlims!(ax, 0.0, 80)
    CairoMakie.ylims!(ax, 0.0, 1.1)
    hlines!(ax, 0.9, color=:black, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, Array(sol)[i, :], color=(:grey, 0.5), linewidth=2)
    end
    lines!(sol.t, Array(sol)[29, :], color=(:red, 0.75), linewidth=3)
    lines!(sol.t, Array(sol)[27, :], color=(:blue, 0.75), linewidth=3)
    f
end
save(projectdir("output/plots/colocalisation/ab_progression.pdf"), f)

begin
    CairoMakie.activate!()
    f = Figure(size = (1200, 350), fontsize=20)
    ax = Axis(f[1,1], xlabel="t / years", ylabel="Concentration", title="Amyloid Progression",
    yticks=0:0.2:1.0, xticks=0:20:150, yticksize=5, ylabelsize=25, xlabelsize=25)
    CairoMakie.xlims!(ax, 0.0, 80)
    CairoMakie.ylims!(ax, 0.0, 1.05)
    hlines!(ax, 0.9, color=:grey, linestyle=:dash, linewidth=3)
    for i in 1:36
        lines!(sol.t, Array(sol)[i, :], color=(:grey, 0.5), linewidth=2)
    end
    lines!(sol.t, Array(sol)[29, :], color=(:red, 0.75), linewidth=3)
    lines!(sol.t, Array(sol)[27, :], color=(:blue, 0.75), linewidth=3)

    ax = Axis(f[1,2], xlabel="t / years", ylabel="Concentration", title="Tau Progression",
              yticks=0:0.2:1.0, xticks=0:20:100, yticksize=5, ylabelsize=25, xlabelsize=25)
    hideydecorations!(ax, grid=false, ticks=false)
    CairoMakie.xlims!(ax, 0.0, 80)
    CairoMakie.ylims!(ax, 0.0, 1.05)
    hlines!(ax, 0.1, color=:grey, linestyle=:dash, linewidth=3)
    for i in 1:36
        lines!(sol.t, Array(sol)[36 + i, :], color=(:grey, 0.5), linewidth=2)
    end
    lines!(sol.t, Array(sol)[36 + 29, :], color=(:red, 0.75), linewidth=3)
    lines!(sol.t, Array(sol)[36 + 27, :], color=(:blue, 0.75), linewidth=3)
    f
end
save(projectdir("output/plots/colocalisation/ab_tau_progression.pdf"), f)
# --------------------------------------------------------------------------------
# Colocalisation
# --------------------------------------------------------------------------------
right_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/colocalisation-order-right.csv"), DataFrame)
left_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/colocalisation-order-left.csv"), DataFrame)

begin
    GLMakie.activate!()
    cmap = reverse(ColorSchemes.RdYlBu)
    f = Figure(size = (600, 500))
    ax = Axis3(f[1,1:5], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, 
             reverse(collect(range(0, 1, 36))), cmap)

    ax = Axis3(f[2,1:5], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, 
              reverse(collect(range(0, 1, 36))), cmap)
    plot_roi!(get_node_id.(left_subcortex), ones(5) .* 0.75, ColorSchemes.Greys)

    ax = Axis3(f[1,6:10], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, 
            reverse(collect(range(0, 1, 36))), cmap)

    ax = Axis3(f[2,6:10], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, 
            reverse(collect(range(0, 1, 36))), cmap)
    plot_roi!(get_node_id.(right_subcortex), ones(5) .* 0.75, ColorSchemes.Greys)
    Colorbar(f[3, 2:9], colormap=reverse(cmap), limits=(1,36), 
             ticks=([1, 36], ["First", "Last"]), ticklabelsize=20, ticksize=10, labelpadding=-20,
             label="Order of colocalisation", vertical = false, flipaxis = false, labelsize=25)
    # f
end
save(projectdir("output/plots/colocalisation/colocalisation-order.jpeg"), f)

right_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/colocalisation-prob-right.csv"), DataFrame)
left_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/colocalisation-prob-left.csv"), DataFrame)
begin
    GLMakie.activate!()
    cmap = ColorSchemes.viridis
    f = Figure(size = (600, 500))
    ax = Axis3(f[1,1:5], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, max_norm(left_df.Seed_prob), cmap)
    
    ax = Axis3(f[2,1:5], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, max_norm(left_df.Seed_prob), cmap)
    plot_roi!(get_node_id.(left_subcortex), ones(5) .* 0.75, ColorSchemes.Greys)

    ax = Axis3(f[1,6:10], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, max_norm(right_df.Seed_prob), cmap)
    
    ax = Axis3(f[2,6:10], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, max_norm(right_df.Seed_prob), cmap)
    plot_roi!(get_node_id.(right_subcortex), ones(5) .* 0.75, ColorSchemes.Greys)

    Colorbar(f[3, 2:9], colormap=cmap, limits=(0, 0.4), ticks=[0,0.4],
              ticklabelsize=20, ticksize=10, vertical = false, flipaxis = false, tellwidth=false,
              label="Probability of colocalisation", labelsize=25, labelpadding=-20)
    
end
save(projectdir("output/plots/colocalisation/colocalisation-prob.jpeg"), f)


using DifferentialEquations

inits = rand(72)

function weighted_diffusion(du, u, p, t; L = L )
    for i in 1:72
        du[i] = -p[1] * (1 / vi[i]) * sum(L[i,:] .* vi .* u)
    end
end

function diffusion(du, u, p, t; L = L )
    for i in 1:72
        du[i] = -p[1] * sum(L[:,i] .* u)
    end
end

function diffusion(du, u, p, t; L = L2)
    du .= -p[1] * L * u
end
L2 = inv(diagm(vi)) * L * diagm(vi)

function fkpp(du, u, p, t; L = L2)
    du .= -p[1] * L * u .+ p[2] .* vi .* u .* ( 1 .- u)
end

e = eigen(Array(L2))
using Connectomes: plot_roi
plot_roi(get_node_id.(cortex), max_norm(abs.(e.vectors[:,1])), ColorSchemes.viridis)

inits = zeros(72)
inits[[27, 63]] .= 0.1
sol = solve(ODEProblem(fkpp, inits, (0.0,15.0), [0.1, 0.5]), Tsit5())
f = Plots.plot(sol, labels=false)

f
Plots.scatter(sol[end])