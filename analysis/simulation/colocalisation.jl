using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model_hemisphere, simulate,
                                    load_ab_params, load_tau_params, conc,
                                    make_atn_prob_func, atn_output_func
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names,
                                    get_braak_regions

using Connectomes: get_hemisphere, plot_roi!, get_node_id, get_lobe
using Colors, ColorSchemes, GLMakie
using DifferentialEquations
# --------------------------------------------------------------------------------
# Connectome and Data
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c) 
cortex = get_parcellation() |> get_cortex 
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

# --------------------------------------------------------------------------------
# Amyloid data
# --------------------------------------------------------------------------------
ab_suvr = calc_suvr.(fbb_data)
normalise!(ab_suvr, fbb_u0, fbb_ui)
ab_conc = map(x -> conc.(x, fbb_u0, fbb_ui), ab_suvr)
ab_inits = [d[:,1] for d in fbb_conc]

mean_ab_init = mean(ab_inits)[1:36]

max_norm(c) =  c ./ maximum(c);

begin
    GLMakie.activate!()
    cmap = ColorSchemes.viridis
    val = mean_ab_init[1:36]
    f = Figure(size = (750, 350), figure_padding = 20, fontsize=25)
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
save(projectdir("output/plots/colocalisation/mean_ab_aptn.jpeg"), f)

# --------------------------------------------------------------------------------
# Tau data
# --------------------------------------------------------------------------------
tau_suvr = calc_suvr.(tau_data)
normalise!(tau_suvr, v0, vi)
tau_conc = map(x -> conc.(x, v0, vi), tau_suvr)
tau_inits = [d[:,1] for d in tau_conc]

mean_tau_init = mean(tau_inits)[1:36]
filtered_tau_idx = findall(x -> x < 0.035, mean_tau_init)
mean_tau_init[filtered_tau_idx] .= 0

using CairoMakie; CairoMakie.activate!()
scatter(mean_tau_init)

begin
    GLMakie.activate!()
    cmap = ColorSchemes.viridis
    val = mean_tau_init[1:36]
    f = Figure(size = (750, 350), figure_padding = 20, fontsize=25)
    ax = Axis3(f[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex), max_norm(val) , cmap)
    
    ax = Axis3(f[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex), max_norm(val), cmap)
    Colorbar(f[2, 1:2], colormap=cmap, limits=(0,0.5), ticks=0:0.1:0.5,
             ticklabelsize=20, ticksize=10, label="Concentration", vertical = false, flipaxis = false)
    f
end

# --------------------------------------------------------------------------------
# Modelling!
# --------------------------------------------------------------------------------
fbb_atn_model = make_scaled_atn_model_hemisphere((fbb_ui .- fbb_u0)[1:36], (part .- v0)[1:36], L[1:36,1:36])

inits = [mean_ab_init; mean_tau_init; zeros(36)]
params = meanpst[:Am_a, :mean], meanpst[:Pm_t, :mean], meanpst[:Am_t, :mean], meanpst[:β, :mean], meanpst[:Em, :mean]

sol = simulate(fbb_atn_model, inits, (0, 200), params)

begin
    CairoMakie.activate!()
    f = Figure(size = (500, 350), figure_padding = 20, fontsize=20)
    ax = Axis(f[1,1], xlabel="t / years", ylabel="Concentration", title="Amyloid Progression",
              yticks=0:0.2:1.0, xticks=0:20:150, yticksize=10)
    CairoMakie.xlims!(ax, 0.0, 150)
    CairoMakie.ylims!(ax, 0.0, 1.1)
    hlines!(ax, 0.9, color=:black, linestyle=:dash)
    for i in 1:36
        lines!(sol.t, Array(sol)[i, :], color=(:grey, 0.5), linewidth=2)
    end
    lines!(sol.t, Array(sol)[29, :], color=(:red, 0.75), linewidth=3)
    lines!(sol.t, Array(sol)[27, :], color=(:blue, 0.75), linewidth=3)
    f
end

begin
    f = Figure(size = (500, 350), fontsize=20)
    ax = Axis(f[1,1], xlabel="t / years", ylabel="Concentration",
              yticks=0:0.2:1.0, xticks=0:20:100, yticksize=10)
    CairoMakie.xlims!(ax, 0.0, 80)
    CairoMakie.ylims!(ax, 0.0, 1.1)
    hlines!(ax, 0.1, color=:black, linestyle=:dash)
    for i in 1:35
        lines!(sol.t, Array(sol)[36 + i, :], color=(:grey, 0.5), linewidth=2)
    end
    lines!(sol.t, Array(sol)[36 + 29, :], color=(:red, 0.75), linewidth=3)
    lines!(sol.t, Array(sol)[36 + 27, :], color=(:blue, 0.75), linewidth=3)
    f
end

ab_sol = Array(sol)[1:36,:]
tau_sol = Array(sol)[37:72,:]

tau_seed = findall(x -> x >= 0.1, tau_sol)
ab_idx = findall(x -> x >= 0.90, ab_sol[tau_seed])
arrival_order = tau_seed[ab_idx]
arrival_regions = unique([a[1] for a in arrival_order])

arrival_idx = arrival_order[[findfirst(x -> x[1] == ar, arrival_order) for ar in arrival_regions]]
arrival_ts_idx = [a[2] for a in arrival_idx]
df = DataFrame(Order = collect(1:36), Region = get_label.(right_cortex)[arrival_regions], Time = sol.t[arrival_ts_idx])

begin
    GLMakie.activate!()
    cmap = ColorSchemes.viridis
    f = Figure(size = (750, 400))
    ax = Axis3(f[1,1:5], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex)[arrival_regions], 
              reverse(collect(range(0, 1, length(arrival_regions)))), cmap)
    ax = Axis3(f[1,6:10], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex)[arrival_regions], 
    reverse(collect(range(0, 1, length(arrival_regions)))), cmap)
    Colorbar(f[2, :], colormap=reverse(cmap), limits=(1,35), 
             ticks=[1, 35], ticklabelsize=20, ticksize=10, label="Order of colocalisation",
             vertical = false, flipaxis = false, labelsize=20)
    f
end

sols = [simulate(fbb_atn_model, inits, (0, 200), params) for params in zip( vec(pst[:Am_a]), vec(pst[:Pm_t]), vec(pst[:Am_t]), vec(pst[:β]), vec(pst[:Em]))];

function find_seed(esol, tau_cutoff, ab_cutoff)
    seed_idx = Vector{Int64}()
    for (i, sol) in enumerate(esol)
        ab_sol = Array(sol)[1:36, :]
        tau_sol = Array(sol)[37:72, :]
        tau_seed = findall(x -> x >= tau_cutoff, tau_sol)
        if length(tau_seed) > 0
            seed = findfirst(x -> x >= ab_cutoff, ab_sol[tau_seed])
            if seed !== nothing
                push!(seed_idx, tau_seed[seed][1])
            end
        end
    end
    seed_idx
end

seed_idx = find_seed(sols, 0.1, 0.9)

seed_count = zeros(36)
seed_count[unique(seed_idx)] .= [count(==(i), seed_idx) for i in unique(seed_idx)]
seed_prob = seed_count ./ length(seed_idx)
init_seed_idx = findall(x -> x > 0, seed_prob)

seed_prob[init_seed_idx]
df = DataFrame(Seed = get_label.(right_cortex)[init_seed_idx], seed_prob = seed_prob[init_seed_idx], lobe=get_lobe.(right_cortex)[init_seed_idx])

begin
    GLMakie.activate!()
    f = Figure(size = (450, 800))
    
    # seed_prob = (seed_prob .+ [seed_prob[36:70]; seed_prob[1:35]]) ./ 2
    ax = Axis3(f[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex), max_norm(seed_prob), cmap)
    
    ax = Axis3(f[2,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(get_node_id.(right_cortex), max_norm(seed_prob), cmap)
    Colorbar(f[3, 1], colormap=cmap, limits=(0, 0.3), ticks=[0,0.3],
              ticklabelsize=20, ticksize=10, vertical = false, flipaxis = false, tellwidth=false,
              label="Probability of colocalisation", labelsize=20)
    f
end