using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, simulate,
                                    load_ab_params, load_tau_params, conc, make_scaled_atn_model_hemisphere
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names,
                                    get_braak_regions

using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise

using Connectomes: get_hemisphere, plot_roi!, get_node_id, get_lobe, laplacian_matrix, get_label, plot_roi
using Colors, ColorSchemes, GLMakie
using DifferentialEquations
using CSV, DataFrames, DrWatson
using ADNIDatasets
using Statistics
using DelimitedFiles
using Serialization
using Turing
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
left_cortex = filter(x -> get_hemisphere(x) == "left", cortex)
# Amyloid data 
_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 && x.NEO_Status == 0, tau_data_df);
_tau_data = ADNIDataset(tau_data_df, dktnames; min_scans=1)
tau_subs = get_id.(_tau_data)
tau_cutoffs = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-1std.csv")) |> vec

tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer 
                    && x.AMYLOID_STATUS_COMPOSITE_REF == 1 && x.RID ∈ tau_subs 
                    && x.CENTILOIDS < 70, _ab_data_df);
mean(fbb_data_df.CENTILOIDS)
amy_pos_init_idx = [findfirst(isequal(id), _ab_data_df.RID) for id in unique(fbb_data_df.RID)]
pos_centiloids = mean(_ab_data_df[amy_pos_init_idx, :].CENTILOIDS)
pos_centiloids_st = std(_ab_data_df[amy_pos_init_idx, :].CENTILOIDS)
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF")

tau_data = filter(x -> get_id(x) ∈ get_id.(fbb_data), _tau_data)

tau_cutoffs = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-1std.csv")) |> vec
# pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-sepbeta-1x1000.jls"));
pst = chainscat([deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-random-beta-lognormal-4x1000-$i.jls")) for i in 1:4]...)

meanpst = mean(pst)
# --------------------------------------------------------------------------------
# Amyloid data
# --------------------------------------------------------------------------------
ab_suvr = calc_suvr.(fbb_data)
normalise!(ab_suvr, fbb_u0, fbb_ui)
ab_conc = map(x -> conc.(x, fbb_u0, fbb_ui), ab_suvr)
ab_inits = [d[:,1] for d in ab_conc]

_mean_ab_init = mean(ab_inits)
_mean_ab_init_sym = (_mean_ab_init[1:36] .+ _mean_ab_init[37:end]) ./ 2
mean_ab_init = [_mean_ab_init_sym; _mean_ab_init_sym]

max_norm(c) =  c ./ maximum(c);

tau_suvr = calc_suvr.(tau_data)
vi = part .+ (meanpst["β_fbb",:mean].* (fbb_ui .- fbb_u0))
normalise!(tau_suvr, v0, vi)
tau_conc = map(x -> conc.(x, v0, vi), tau_suvr)
tau_inits = [d[:,1] for d in tau_conc]

_mean_tau_init = mean(tau_inits)
idx = _mean_tau_init .< conc.(tau_cutoffs, v0, vi)
_mean_tau_init[idx] .= 0
_mean_tau_init_sym = mean.(zip(_mean_tau_init[1:36], _mean_tau_init[37:end]))
mean_tau_init = [_mean_tau_init_sym; _mean_tau_init_sym]
scatter(mean_tau_init)
# --------------------------------------------------------------------------------
# Modelling!
# --------------------------------------------------------------------------------
hem_idx = 1:72
atn_model = make_scaled_atn_model((fbb_ui .- fbb_u0)[hem_idx], (part .- v0)[hem_idx], L[hem_idx,hem_idx])

inits = [mean_ab_init[hem_idx]; mean_tau_init[hem_idx]; zeros(72)]

amyloid_production = mean([meanpst["α_a[$i]", :mean] for i in 1:18])
tau_transport = mean([meanpst["ρ_t[$i]", :mean] for i in 1:18])
tau_production = mean([meanpst["α_t[$i]", :mean] for i in 1:18])
coupling = meanpst["β_fbb", :mean]
atrophy = mean([meanpst["η[$i]", :mean] for i in 1:18])
params = [amyloid_production, tau_transport, tau_production, coupling, atrophy]

sol = solve(ODEProblem(atn_model, inits, (0, 80), params), Tsit5(), saveat=0.1)
soldt = solve(ODEProblem(atn_model, inits, (0, 80), params), Tsit5(), abstol=1e-12, reltol=1e-12)
f = plot(sol, idxs=73:144)
plot!(sol,idxs=73:108)
plot!(sol,idxs=109:144, color=:red)
f
absol = Array(sol[1:72,:])
tausol = Array(sol[73:144,:])

ab_threshold = readdlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/ab-thresholds.csv")) |> vec
tau_threshold = readdlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/tau-thresholds.csv")) |> vec

d1 = reduce(hcat, [soldt(t, Val{1}) for t in 0:0.1:80])
d2 = reduce(hcat, [soldt(t, Val{2}) for t in 0:0.1:80])
d3 = reduce(hcat, [soldt(t, Val{3}) for t in 0:0.1:80])
ts = 0:0.1:80

# ----------------------------------------------------------------------------------------------------
# ALL
# ----------------------------------------------------------------------------------------------------
begin
    GLMakie.activate!()
    cmap = ColorSchemes.viridis
    scmap = ColorSchemes.Greys

    f = Figure(size = (1600, 900))

    g1 = f[1,1] = GridLayout()
    g2 = f[2,1] = GridLayout()
    g3 = f[1,2:3] = GridLayout()
    g4 = f[2,2:3] = GridLayout()

    # AB
    # ax = Axis(g1[1,1], xlabel="t / years", ylabel="Conc.", title="",titlefont=:regular,titlesize=25, 
    # yticks=0.2:0.4:1.0, xticks=0:10:150, ylabelsize=25, xlabelsize=25, xticklabelsize=25, yticklabelsize=25,)
    # hidexdecorations!(ax, grid=false, ticks=false)
    # xlims!(ax, 0.0, 50); ylims!(ax, 0.2, 1.05)
    it_idx = 65
    ec_idx = 63
    # plot!(sol, idxs=it_idx)
    # # linesegment(ts[argmin(d2[it_idx,:])])
    # linesegments!([(ts[argmax(d1[it_idx,:])], 0.2), (ts[argmax(d1[it_idx,:])], 0.5)], color=:grey, linestyle=:dash, linewidth=2)
    # linesegments!([(0.0, 0.5),  (ts[argmax(d1[it_idx,:])], 0.5)], color=:grey, linestyle=:dash, linewidth=2)

    ax = Axis(g1[1,1], ylabel=L"dx/dt", xlabel="t / years", yticks=[0.], ytickformat="{:.1f}", xlabelsize=25, ylabelsize=25, yticklabelsize=25, xticklabelsize=25)
    hidexdecorations!(ax, grid=false, ticks=false)
    xlims!(ax, 0.0, 50); 
    vlines!(ts[argmax(d1[it_idx,:])], color=:grey, linestyle=:dash, linewidth=2)
    lines!(ts, d1[it_idx,:], color=:blue)

    ax = Axis(g1[2,1], ylabel="Conc.", xlabel="t / years", yticks=0.:0.2:1.0, ytickformat="{:.1f}", xlabelsize=25, ylabelsize=25, yticklabelsize=25, xticklabelsize=25)
    xlims!(ax, 0.0, 50); ylims!(ax, 0., 1.05)
    hlines!(ax, 0.5, color=:grey, linestyle=:dash, linewidth=2)
    # vlines!(ts[argmin(d2[it_idx,:])], color=:grey, linestyle=:dash, linewidth=2)
    for i in 37:72
        lines!(sol.t, Array(sol)[i, :], color=(:grey, 0.5), linewidth=2)
    end
    lines!(sol.t, Array(sol)[it_idx, :], color=(:blue, 0.75), linewidth=2, label="Inferior temporal")
    lines!(sol.t, Array(sol)[ec_idx, :], color=(:red, 0.75), linewidth=2, label="Entorhinal cortex")
    vlines!(ts[argmax(d1[it_idx,:])], color=:grey, linestyle=:dash, linewidth=2)
    axislegend(ax, position=:rb, labelsize=20)
    rowgap!(g1, 7)
    rowsize!(g1, 1, 120)
    
    # tau
    # ax = Axis(g2[1,1], xlabel="t / years", ylabel="Conc.", title="",titlefont=:regular,titlesize=25, 
    # yticks=0.:0.5:1.0, xticks=0:10:150, ylabelsize=25, xlabelsize=25, xticklabelsize=25, yticklabelsize=25,)
    # hidexdecorations!(ax, grid=false, ticks=false)
    # xlims!(ax, 0.0, 50); ylims!(ax, 0.0, 1.05)
    # plot!(sol, idxs=it_idx+72)
    # linesegments!([(ts[argmax(d3[it_idx+72,:])], 0.0), (ts[argmax(d3[it_idx+72,:])], tau_threshold[it_idx])], color=:grey, linestyle=:dash, linewidth=2)
    # linesegments!([(0.0, tau_threshold[it_idx]),  (ts[argmax(d3[it_idx+72,:])], tau_threshold[it_idx])], color=:grey, linestyle=:dash, linewidth=2)

    ax = Axis(g2[1,1], ylabel=L"d^{3}x/dt^{3}", xlabel="t / years", yticks=[0.], ytickformat="{:.1f}", xlabelsize=25, ylabelsize=25, yticklabelsize=25, xticklabelsize=25)
    hidexdecorations!(ax, grid=false, ticks=false)
    xlims!(ax, 0.0, 50); 
    lines!(ts, d3[it_idx+72,:], color=:blue)
    vlines!(ts[argmax(d3[it_idx+72,:])], color=:grey, linestyle=:dash, linewidth=2)

    ax = Axis(g2[2,1], ylabel="Conc.", xlabel="t / years", yticks=0.:0.2:1.0, ytickformat="{:.1f}", xlabelsize=25, ylabelsize=25, yticklabelsize=25, xticklabelsize=25)
    xlims!(ax, 0.0, 50); ylims!(ax, 0., 1.05)
    hlines!(ax, tau_threshold[it_idx], color=:grey, linestyle=:dash, linewidth=2)
    for i in 37:72
        lines!(sol.t, Array(sol)[72 + i, :], color=(:grey, 0.5), linewidth=2)
    end
    lines!(sol.t, Array(sol)[72 + it_idx, :], color=(:blue, 0.75), linewidth=2, label="Inferior temporal")
    lines!(sol.t, Array(sol)[72 + ec_idx, :], color=(:red, 0.75), linewidth=2, label="Entorhinal cortex")
    axislegend(ax, position=:rb, labelsize=20)
    vlines!(ts[argmax(d3[it_idx+72,:])], color=:grey, linestyle=:dash, linewidth=2)
    rowgap!(g2, 7)
    rowsize!(g2, 1, 120)

    # coloc Order
    right_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-order-right.csv"), DataFrame)
    left_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-order-left.csv"), DataFrame)
    cmap = reverse(ColorSchemes.RdYlBu)

    Label(g3[1,1:2], "ADNI", fontsize=25, )
    ax = Axis3(g3[2,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, reverse(collect(range(0, 1, 36))), cmap)
    plot_roi!(get_node_id.(left_subcortex), fill(0.5, 5), scmap)

    ax = Axis3(g3[3,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, reverse(collect(range(0, 1, 36))), cmap)
    plot_roi!(get_node_id.(left_subcortex), fill(0.5, 5), scmap)

    ax = Axis3(g3[2,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, reverse(collect(range(0, 1, 36))), cmap)
    plot_roi!(get_node_id.(right_subcortex), fill(0.5, 5), scmap)

    ax = Axis3(g3[3,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, reverse(collect(range(0, 1, 36))), cmap)
    plot_roi!(get_node_id.(right_subcortex), fill(0.5, 5), scmap)

    right_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation-bf/05-tau-thresholds-random/colocalisation-inits-order-right.csv"), DataFrame)
    left_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation-bf/05-tau-thresholds-random/colocalisation-inits-order-left.csv"), DataFrame)
    Label(g3[1,3:4], "BF2", fontsize=25, )

    ax = Axis3(g3[2,3], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, reverse(collect(range(0, 1, 36))), cmap)
    plot_roi!(get_node_id.(left_subcortex), fill(0.5, 5), scmap)

    ax = Axis3(g3[3,3], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, reverse(collect(range(0, 1, 36))), cmap)
    plot_roi!(get_node_id.(left_subcortex), fill(0.5, 5), scmap)

    ax = Axis3(g3[2,4], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, reverse(collect(range(0, 1, 36))), cmap)
    plot_roi!(get_node_id.(right_subcortex), fill(0.5, 5), scmap)
    
    ax = Axis3(g3[3,4], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, reverse(collect(range(0, 1, 36))), cmap)
    plot_roi!(get_node_id.(right_subcortex), fill(0.5, 5), scmap)
    
    cb = Colorbar(g3[2:3, 5], colormap=reverse(cmap), limits=(1,36), 
             ticks=([1, 36], ["First", "Last"]), ticklabelsize=25, ticksize=10, labelpadding=-40,
             label="Colocalisation \n Order", vertical = true, flipaxis = true, labelsize=25)


    # coloc prob
    right_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-prob-right.csv"), DataFrame)
    left_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-prob-left.csv"), DataFrame)
    max_prob = maximum([left_df.Seed_prob; right_df.Seed_prob])
    cmap = ColorSchemes.viridis

    Label(g4[1,1:2], "ADNI", fontsize=25, )
    ax = Axis3(g4[2,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, left_df.Seed_prob ./ max_prob, cmap)
    plot_roi!(get_node_id.(left_subcortex), fill(0.5, 5), scmap)

    ax = Axis3(g4[3,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, left_df.Seed_prob ./ max_prob, cmap)
    plot_roi!(get_node_id.(left_subcortex), fill(0.5, 5), scmap)

    ax = Axis3(g4[2,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, right_df.Seed_prob ./ max_prob, cmap)
    plot_roi!(get_node_id.(right_subcortex), fill(0.5, 5), scmap)
    
    ax = Axis3(g4[3,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, right_df.Seed_prob ./ max_prob, cmap)
    plot_roi!(get_node_id.(right_subcortex), fill(0.5, 5), scmap)
    
    right_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation-bf/05-tau-thresholds-random/colocalisation-inits-prob-right.csv"), DataFrame)
    left_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation-bf/05-tau-thresholds-random/colocalisation-inits-prob-left.csv"), DataFrame)

    Label(g4[1,3:4], "BF2", fontsize=25, )
    ax = Axis3(g4[2,3], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, left_df.Seed_prob ./ max_prob, cmap)
    plot_roi!(get_node_id.(left_subcortex), fill(0.5, 5), scmap)

    ax = Axis3(g4[3,3], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, left_df.Seed_prob ./ max_prob, cmap)
    plot_roi!(get_node_id.(left_subcortex), fill(0.5, 5), scmap)

    ax = Axis3(g4[2,4], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, right_df.Seed_prob ./ max_prob, cmap)
    plot_roi!(get_node_id.(right_subcortex), fill(0.5, 5), scmap)

    ax = Axis3(g4[3,4], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, right_df.Seed_prob ./ max_prob, cmap)
    plot_roi!(get_node_id.(right_subcortex), fill(0.5, 5), scmap)

    cb = Colorbar(g4[2:3, 5], colormap=cmap, limits=(0,0.6), 
             ticks=[0.0, 0.6], ticklabelsize=25, ticksize=10, labelpadding=-27.5,
             label="Initial Colocalisation \n Probability", vertical = true, flipaxis = true, labelsize=25)

    Label(g1[1, 1, TopLeft()], "A", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    Label(g2[1, 1, TopLeft()], "B", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    Label(g3[1, 1, TopLeft()], "C", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    Label(g4[1, 1, TopLeft()], "D", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    
    
end
save(projectdir("output/plots/colocalisation/colocalisation-05-tau-thresholds-random-all.jpeg"), f)


using PrettyTables, CSV, DataFrames, DrWatson
left_df = filter( x -> x.Seed_prob > 0.0, CSV.read(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-prob-left.csv"), DataFrame))
right_df = filter( x -> x.Seed_prob > 0.0, CSV.read(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-prob-right.csv"), DataFrame))

df = DataFrame(
          Right_Seed = [reverse(right_df.Seed); [""]],
          Right_prob = [reverse(right_df.Seed_prob); [""]],
          Left_seed = reverse(left_df.Seed),
          left_prob = reverse(left_df.Seed_prob)
          )
formatter = (v, i, j) -> round(v, digits = 3);

pretty_table(df; formatters = ft_printf("%5.3f"), backend=Val(:latex))

left_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-order-left.csv"), DataFrame)
right_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/05-tau-thresholds-random/colocalisation-inits-order-right.csv"), DataFrame)

df = DataFrame(
          Right_Seed = right_df.Region ,
          Right_prob = right_df.Coloc_time,
          Left_seed = left_df.Region,
          left_prob = left_df.Coloc_time
          )
formatter = (v, i, j) -> round(v, digits = 2);

pretty_table(df; formatters = ft_printf("%5.2f"), backend=Val(:latex))

using PrettyTables, CSV, DataFrames, DrWatson
left_df = filter( x -> x.Seed_prob > 0.0, CSV.read(projectdir("output/analysis-derivatives/colocalisation-bf/05-tau-thresholds-random/colocalisation-inits-prob-left.csv"), DataFrame))
right_df = filter( x -> x.Seed_prob > 0.0, CSV.read(projectdir("output/analysis-derivatives/colocalisation-bf/05-tau-thresholds-random/colocalisation-inits-prob-right.csv"), DataFrame))

df = DataFrame(
          Right_Seed = reverse(right_df.Seed),
          Right_prob = reverse(right_df.Seed_prob),
          Left_seed = reverse(left_df.Seed),
          left_prob = reverse(left_df.Seed_prob)
          )
formatter = (v, i, j) -> round(v, digits = 3);

pretty_table(df; formatters = ft_printf("%5.3f"), backend=Val(:latex))

left_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation-bf/05-tau-thresholds-random/colocalisation-inits-order-left.csv"), DataFrame)
right_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation-bf/05-tau-thresholds-random/colocalisation-inits-order-right.csv"), DataFrame)

df = DataFrame(
          Right_Seed = right_df.Region ,
          Right_prob = right_df.Coloc_time,
          Left_seed = left_df.Region,
          left_prob = left_df.Coloc_time
          )
formatter = (v, i, j) -> round(v, digits = 2);

pretty_table(df; formatters = ft_printf("%5.2f"), backend=Val(:latex))