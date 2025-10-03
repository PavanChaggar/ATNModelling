using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, simulate,
                                    load_ab_params, load_tau_params, conc, make_scaled_atn_model_hemisphere
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names,
                                    get_braak_regions

using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise

using Connectomes: get_hemisphere, plot_roi!, get_node_id, get_lobe, laplacian_matrix, get_label
using Colors, ColorSchemes, GLMakie
using DifferentialEquations
using CSV, DataFrames, DrWatson
using ADNIDatasets
using Statistics
using DelimitedFiles
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
left_cortex = filter(x -> get_hemisphere(x) == "left", cortex)
# Amyloid data 
_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 && x.NEO_Status == 0, tau_data_df);
_tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=1)
tau_subs = get_id.(_tau_data)
tau_cutoffs = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-2std.csv")) |> vec

tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer && x.AMYLOID_STATUS_COMPOSITE_REF == 1 && x.RID ∈ tau_subs && x.CENTILOIDS < 67, _ab_data_df);
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF")

tau_data = filter(x -> get_id(x) ∈ get_id.(fbb_data), _tau_data)

tau_cutoffs = readdlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-1std.csv")) |> vec
pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-sepbeta-1x1000.jls"));
pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-random-beta-lognormal-1x1000.jls"));

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
# tau_cutoffs = fill(0.05, 72)
idx = _mean_tau_init .< conc.(tau_cutoffs, v0, vi)
# idx = _mean_tau_init .< tau_cutoffs
_mean_tau_init[idx] .= 0
_mean_tau_init_sym = maximum.(zip(_mean_tau_init[1:36], _mean_tau_init[37:end]))
mean_tau_init = [_mean_tau_init_sym; _mean_tau_init_sym]
scatter(mean_tau_init)
# --------------------------------------------------------------------------------
# Modelling!
# --------------------------------------------------------------------------------
hem_idx = 1:72
atn_model = make_scaled_atn_model((fbb_ui .- fbb_u0)[hem_idx], (part .- v0)[hem_idx], L[hem_idx,hem_idx])

inits = [mean_ab_init[hem_idx]; mean_tau_init[hem_idx]; zeros(72)]
params = [0.34, 0.05, 0.12, meanpst["β_fbb",:mean], 0.14]
# params = meanpst[:Am_a, :mean], meanpst[:Pm_t, :mean], meanpst[:Am_t, :mean], 3.2258211441306877, meanpst[:Em, :mean]

sol = solve(ODEProblem(atn_model, inits, (0, 80), params), Tsit5(), saveat=0.1)
soldt = solve(ODEProblem(atn_model, inits, (0, 80), params), Tsit5(), abstol=1e-12, reltol=1e-12)

absol = Array(sol[1:72,:])
tausol = Array(sol[73:144,:])

# tau_seed = findall(x -> x >= 0.09, tausol)
# tau_seed_idx = zeros(36, size(sol,2))
# tau_seed_idx[tau_seed] .= 1.0
tau_seed_idx = tausol .>= tau_threshold[1:72]
# heatmap(tau_seed_idx)
cortical_idx = setdiff(collect(1:36), [63, 36])
tau_t = findfirst(x -> x > 0, tau_seed_idx)
sol.t[tau_t[2]]

# ab_seed = findall(x -> x >= 0.79, absol)
# ab_seed_idx = zeros(36, size(sol,2))
# ab_seed_idx[ab_seed] .= 1.0
ab_seed_idx = absol .>= ab_threshold
# heatmap(ab_seed_idx)

ab_tau_coloc = tau_seed_idx .* ab_seed_idx
# heatmap(ab_tau_coloc)
coloc_t = findall(x -> x > 0, sum(ab_tau_coloc, dims=1))
coloc_node = findall(x -> x > 0, ab_tau_coloc[:, coloc_t[1][2]])
dktnames[coloc_node][1]
sol.t[coloc_t[1][2]]

d1 = reduce(hcat, [soldt(t, Val{1}) for t in 0:0.1:80])
d2 = reduce(hcat, [soldt(t, Val{2}) for t in 0:0.1:80])
d3 = reduce(hcat, [soldt(t, Val{3}) for t in 0:0.1:80])
ts = 0:0.1:80

ab_threshold = readdlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/ab-thresholds.csv"))
tau_threshold = readdlm(projectdir("output/analysis-derivatives/colocalisation/thresholds/tau-thresholds.csv"))
begin
    f = Figure(size=(1000, 600))
    node = 65
    ax = Axis(f[1,1])
    ylims!(ax, 0, 1)
    plot!(sol, idxs=72+node)
    vlines!(ts[argmax(d3[65+72,:])], color=:grey, linestyle=:dash, linewidth=2)
    hlines!(ax, tau_threshold[65], color=:grey, linestyle=:dash, linewidth=2)

    ax = Axis(f[2,1])
    # ylims!(ax, -0.015, 0.015)
    # lines!(0:0.1:80, d2[72+node, :])
    lines!(0:0.1:80, d3[72+node, :])

    vlines!(ts[argmax(d3[65+72,:])], color=:grey, linestyle=:dash, linewidth=2)

    f
end


begin
    f = Figure()
    ax = Axis(f[1,1])
    plot!(sol, idxs=72+ 65)
    plot!(sol, idxs=72+ 62, color=:red)
    # vlines!(ts[argmin(d2[65,:])])

    # ax = Axis(f[2, 1])
    # scatter!(ts, d1[65,:])
    # scatter!(ts, d2[65,:])
    # scatter!(ts, d3[65,:])
    # vlines!(ts[argmin(d2[65,:])])
    f
end

begin
    GLMakie.activate!()
    f = Figure(size = (1000, 1000))

    g1 = f[1,1:2] = GridLayout()
    g2 = f[2,1:2] = GridLayout()
    g3 = f[3,1] = GridLayout()
    g4 = f[3,2] = GridLayout()
    cmap = ColorSchemes.viridis

    # Initial conditions 
    # ab_val = mean_ab_init[1:36]
    # ax = Axis3(g1[2,3], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    # hidedecorations!(ax); hidespines!(ax)
    # plot_roi!(get_node_id.(right_cortex), max_norm(ab_val) , cmap)
    # # ax.alignmode = Mixed(top = 20)

    # ax = Axis3(g1[2,4], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    # hidedecorations!(ax); hidespines!(ax)
    # plot_roi!(get_node_id.(right_cortex), max_norm(ab_val), cmap)
    # # ax.alignmode = Mixed(top = 20)

    # tau_val = mean_tau_init[1:36]
    # ax = Axis3(g1[2,5], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    # hidedecorations!(ax); hidespines!(ax)
    # plot_roi!(get_node_id.(right_cortex), tau_val ./ maximum(ab_val) , cmap)
    # # ax.alignmode = Mixed(top = 20)
    # ax = Axis3(g1[2,6], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(1.0,1.0,1.0,1.0))
    # hidedecorations!(ax); hidespines!(ax)
    # plot_roi!(get_node_id.(right_cortex), tau_val ./ maximum(ab_val), cmap)
    # cb = Colorbar(g1[2, 2], colormap=cmap, limits=(0,0.6), ticks=0:0.2:0.6,
    #          ticklabelsize=25, vertical = true, flipaxis = false, labelsize=25)
    # cb.alignmode = Mixed(left = 0, top = 40, bottom = 40)
    # ax.alignmode = Mixed(top = 20)
    # Label(g1[2, 1], "Concentration", fontsize=25, rotation=pi/2, tellheight=false, padding = (0, 0, 0, 20))
    # Label(g1[1, 3:4], "Aβ", fontsize=25, tellwidth=false, tellheight=true, padding = (0, 0, -50, 0))
    # Label(g1[1, 5:6], "Tau", fontsize=25, tellwidth=false, tellheight=true, padding = (0, 0, -50, 0))

    # la.alignmode = Mixed(top = 30)


    ax = Axis(g1[1,1], xlabel="t / years", ylabel="Conc.", title="Aβ Progression",titlefont=:regular,titlesize=25, 
    yticks=0.2:0.4:1.0, xticks=0:10:150, ylabelsize=25, xlabelsize=25, xticklabelsize=25, yticklabelsize=25,)
    ax.alignmode = Mixed(left = 0, right=0)
    hidexdecorations!(ax, grid=false, ticks=false)
    xlims!(ax, 0.0, 50)
    ylims!(ax, 0.2, 1.05)
    plot!(sol, idxs=65)
    # linesegment(ts[argmin(d2[65,:])])
    linesegments!([(ts[argmin(d2[65,:])], 0.4), (ts[argmin(d2[65,:])], ab_threshold[65])], color=:grey, linestyle=:dash, linewidth=2)
    linesegments!([(0.0, ab_threshold[65]),  (ts[argmin(d2[65,:])], ab_threshold[65])], color=:grey, linestyle=:dash, linewidth=2)

    ax = Axis(g1[2,1], ylabel=L"d^{2}x/dt^{2}", xlabel="t / years", yticks=[0.], ytickformat="{:.1f}", ylabelsize=25, yticklabelsize=25, xticklabelsize=25)
    ax.alignmode = Mixed(left = 0, right=0)
    xlims!(ax, 0.0, 50)
    vlines!(ts[argmin(d2[65,:])], color=:grey, linestyle=:dash, linewidth=2)
    lines!(ts, d2[65,:])
    hidexdecorations!(ax, grid=false, ticks=false)
    # hideydecorations!(ax, grid=false, ticks=false, label=false)

    ax = Axis(g1[1,2], xlabel="t / years", ylabel="Conc.", title="Tau Progression",titlefont=:regular, titlesize=25,
    yticks=0:0.5:1.0, xticks=0:10:150, ylabelsize=25, xlabelsize=25, xticklabelsize=25, yticklabelsize=25)
    ax.alignmode = Mixed(left = 0, right=0)
    hidexdecorations!(ax, grid=false, ticks=false)
    xlims!(ax, 0.0, 50)
    plot!(sol, idxs=65+72)
    linesegments!([(ts[argmax(d3[65+72,:])], 0.0), (ts[argmax(d3[65+72,:])], tau_threshold[65])], color=:grey, linestyle=:dash, linewidth=2)
    linesegments!([(0.0, tau_threshold[65]),  (ts[argmax(d3[65+72,:])], tau_threshold[65])], color=:grey, linestyle=:dash, linewidth=2)

    ax = Axis(g1[2,2], ylabel=L"d^{3}x/dt^{3}", xlabel="t / years", yticks=[0.], ytickformat="{:.1f}", ylabelsize=25, yticklabelsize=25, xticklabelsize=25)
    ax.alignmode = Mixed(left = 0, right=0)
    xlims!(ax, 0.0, 50)
    lines!(ts, d3[65+72,:])
    hidexdecorations!(ax, grid=false, ticks=false)
    rowgap!(g1, 7)
    vlines!(ts[argmax(d3[65+72,:])], color=:grey, linestyle=:dash, linewidth=2)

    # tau progression 
    ax = Axis(g2[1,1], xlabel="t / years", ylabel="Conc.", 
    yticks=0:0.2:1.0, xticks=0:10:150, ylabelsize=25, xlabelsize=25, xticklabelsize=25, yticklabelsize=25, titlesize=25)
    # hideydecorations!(ax, grid=false, ticks=false, ticklabels=false)
    xlims!(ax, 0.0, 50)
    ylims!(ax, 0.0, 1.05)
    hlines!(ax, ab_threshold[65], color=:grey, linestyle=:dash, linewidth=2)
    # vlines!(ts[argmin(d2[65,:])], color=:grey, linestyle=:dash, linewidth=2)
    for i in 1:36
        lines!(sol.t, Array(sol)[i, :], color=(:grey, 0.5), linewidth=2)
    end
    lines!(sol.t, Array(sol)[65, :], color=(:red, 0.75), linewidth=2)
    lines!(sol.t, Array(sol)[63, :], color=(:blue, 0.75), linewidth=2)
    vlines!(ts[argmin(d2[65,:])], color=:grey, linestyle=:dash, linewidth=2)
    ax.alignmode = Mixed(right = 0, left = 0 )
    
    ax = Axis(g2[1,2], xlabel="t / years", ylabel="Conc.", 
    yticks=0:0.2:1.0, xticks=0:10:150, ylabelsize=25, xlabelsize=25, xticklabelsize=25, yticklabelsize=25, titlesize=25)
    # hideydecorations!(ax, grid=false, ticks=false, ticklabels=false)
    xlims!(ax, 0.0, 50)
    ylims!(ax, 0.0, 1.05)
    hlines!(ax, tau_threshold[65], color=:grey, linestyle=:dash, linewidth=2)
    # vlines!(ts[argmax(d3[65+36,:])], color=:grey, linestyle=:dash, linewidth=2)
    for i in 1:36
        lines!(sol.t, Array(sol)[72 + i, :], color=(:grey, 0.5), linewidth=2)
    end
    lines!(sol.t, Array(sol)[72 + 65, :], color=(:red, 0.75), linewidth=2)
    lines!(sol.t, Array(sol)[72 + 63, :], color=(:blue, 0.75), linewidth=2)
    vlines!(ts[argmax(d3[65+72,:])], color=:grey, linestyle=:dash, linewidth=2)

    ax.alignmode = Mixed(right = 0 , left = 0)
    

    # Label(g2[1, 1], "Concentration", fontsize=25, rotation=pi/2, tellheight=false)

    # coloc order 
    right_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/ab-tau-thresholds-random/colocalisation-inits-order-right.csv"), DataFrame)
    left_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/ab-tau-thresholds-random/colocalisation-inits-order-left.csv"), DataFrame)
    # df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/0109/colocalisation-inits-order-all.csv"), DataFrame)
    # left_df = filter(x -> x.Hemisphere == "left", df)
    # right_df = filter(x -> x.Hemisphere == "right", df)

    cmap = reverse(ColorSchemes.RdYlBu)
    ax = Axis3(g3[1,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, 
            reverse(collect(range(0, 1, 36))), cmap)
    ax = Axis3(g3[2,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, 
            reverse(collect(range(0, 1, 36))), cmap)
    plot_roi!(get_node_id.(left_subcortex), ones(5) .* 0.75, ColorSchemes.Greys)

    ax = Axis3(g3[1,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, 
                reverse(collect(range(0, 1, 36))), cmap)

    ax = Axis3(g3[2,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,   protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, 
                reverse(collect(range(0, 1, 36))), cmap)
    plot_roi!(get_node_id.(right_subcortex), ones(5) .* 0.75, ColorSchemes.Greys)
    cb = Colorbar(g3[3, :], colormap=reverse(cmap), limits=(1,36), 
             ticks=([1, 36], ["First", "Last"]), ticklabelsize=25, ticksize=10, labelpadding=-20,
             label="Order \n of colocalisation", vertical = false, flipaxis = false, labelsize=25)
    cb.alignmode = Mixed(left = 50, right = 50 )

    # coloc prob 
    right_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/ab-tau-thresholds-random/colocalisation-inits-prob-right.csv"), DataFrame)
    left_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/ab-tau-thresholds-random/colocalisation-inits-prob-left.csv"), DataFrame)
    df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/ab-tau-thresholds-random/colocalisation-inits-prob-all.csv"), DataFrame)
    # left_df = filter(x -> x.Hemisphere == "left", df)
    # right_df = filter(x -> x.Hemisphere == "right", df)
    max_prob = maximum([left_df.Seed_prob; right_df.Seed_prob])
    cmap = ColorSchemes.viridis
    ax = Axis3(g4[1,1], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, left_df.Seed_prob ./ max_prob, cmap)
    
    ax = Axis3(g4[2,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(left_df.DKTID, left_df.Seed_prob ./ max_prob, cmap)
    plot_roi!(get_node_id.(left_subcortex), ones(5) .* 0.75, ColorSchemes.Greys)

    ax = Axis3(g4[1,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, right_df.Seed_prob ./ max_prob, cmap)
    
    ax = Axis3(g4[2,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi,  protrusions=(0.0,0.0,0.0,0.0))
    hidedecorations!(ax); hidespines!(ax)
    plot_roi!(right_df.DKTID, right_df.Seed_prob ./ max_prob, cmap)
    plot_roi!(get_node_id.(right_subcortex), ones(5) .* 0.75, ColorSchemes.Greys)

    cb = Colorbar(g4[3, :], colormap=cmap, limits=(0, 0.5), ticks=[0,0.5],
              ticklabelsize=25, ticksize=10, vertical = false, flipaxis = false, tellwidth=false,
              label="Probability \n of colocalisation", labelsize=25, labelpadding=-20)
    cb.alignmode = Mixed(left = 50, right = 50 )

    # rowgap!(f.layout, 0, 50)
    rowgap!(f.layout, 1, 7)
    rowgap!(f.layout, 2, 20)
    colgap!(g3, 1, -15)
    colgap!(g4, 1, -15)
    rowsize!(f.layout, 1, 200)
    # rowsize!(f.layout, 2, 200)
    rowsize!(f.layout, 3, 400)


    Label(g1[1, 1, TopLeft()], "A", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    # Label(g2[1, 1, TopLeft()], "B", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    Label(g3[1, 1, TopLeft()], "B", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    Label(g4[1, 1, TopLeft()], "C", fontsize = 26, font = :bold, padding = (0, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    
    f
end
save(projectdir("output/plots/colocalisation/colocalisation-ab-tau-thresholds-random.jpeg"), f)

# late tables 
using PrettyTables, CSV, DataFrames, DrWatson
left_df = filter( x -> x.Seed_prob > 0.0, CSV.read(projectdir("output/analysis-derivatives/colocalisation/007579-random/colocalisation-inits-prob-left.csv"), DataFrame))
right_df = filter( x -> x.Seed_prob > 0.0, CSV.read(projectdir("output/analysis-derivatives/colocalisation/007579-random/colocalisation-inits-prob-right.csv"), DataFrame))

df = DataFrame(
          Right_Seed = [reverse(right_df.Seed);""],
          Right_prob = [reverse(right_df.Seed_prob); 0],
          Left_seed = [reverse(left_df.Seed); [""]],
          left_prob = [reverse(left_df.Seed_prob);[0]]
          )
formatter = (v, i, j) -> round(v, digits = 3);

pretty_table(df; formatters = ft_printf("%5.3f"), backend=Val(:latex))

using PrettyTables, CSV, DataFrames, DrWatson
left_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/007579-random/colocalisation-inits-order-left.csv"), DataFrame)
right_df = CSV.read(projectdir("output/analysis-derivatives/colocalisation/007579-random/colocalisation-inits-order-right.csv"), DataFrame)

df = DataFrame(
          Right_Seed = right_df.Region ,
          Right_prob = right_df.Coloc_time,
          Left_seed = left_df.Region,
          left_prob = left_df.Coloc_time
          )
formatter = (v, i, j) -> round(v, digits = 3);

pretty_table(df; formatters = ft_printf("%5.3f"), backend=Val(:latex))