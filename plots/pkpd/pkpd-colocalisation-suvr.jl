using ATNModelling.SimulationUtils: make_prob, make_atn_pkpd_model, 
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params, conc, make_scaled_atn_pkpd_model
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names, get_distance_laplacian, get_braak_regions
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise

using Connectomes: laplacian_matrix, get_label, get_hemisphere, get_node_id, plot_roi!
using ADNIDatasets: ADNIDataset, get_id, get_dates, get_initial_conditions, calc_suvr, get_vol, get_times
using DrWatson: projectdir, datadir
using CSV, DataFrames
using CairoMakie, Colors, ColorSchemes, GLMakie
using Statistics, SciMLBase
# --------------------------------------------------------------------------------
# Tracer independent data
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params()
parc = get_parcellation() |> get_cortex
cortex = filter(x -> get_hemisphere(x) == "right", parc)

c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c) 
Lh = L[1:36, 1:36]
Ld = get_distance_laplacian()

cingulate = findall(x -> contains(get_label(x), "cingulate"), cortex)
m = zeros(36)
m[cingulate] .= 1
m[[35,36]] .= 1

dktnames = get_parcellation() |> get_cortex |> get_dkt_names

using Serialization
pst = deserialize(projectdir("output/chains/population-atn/pst-samples-harmonised-suvr-sepbeta-1x1000.jls"));
meanpst = mean(pst)
mean([meanpst["α_a[$i]", :mean] for i in 1:18])
mean([meanpst["ρ_t[$i]", :mean] for i in 1:18])
mean([meanpst["α_t[$i]", :mean] for i in 1:18])
mean([meanpst["η[$i]", :mean] for i in 1:18])

mean([meanpst["α_a[$i]", :mean] for i in 1:34])
mean([meanpst["ρ_t[$i]", :mean] for i in 1:34])
mean([meanpst["α_t[$i]", :mean] for i in 1:34])
mean([meanpst["η[$i]", :mean] for i in 1:34])
# --------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------
_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 && x.NEO_Status == 0, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=1)
IDS = unique(tau_pos_df.RID)

tracer="FBB"

fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer && x.RID ∈ IDS, _ab_data_df)
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF")

fbb_suvr = calc_suvr.(fbb_data)
normalise!(fbb_suvr, fbb_u0, fbb_ui)
fbb_conc = map(x -> conc.(x, fbb_u0, fbb_ui), fbb_suvr)
fbb_inits = [d[:,1] for d in fbb_suvr]
mean_fbb_init = mean(fbb_inits)[1:36]
println("ab concentration = $(mean_fbb_init[29])")

fbb_tau_suvr = calc_suvr.(tau_data)
vi = part .+ (3.2258211441306877 .* (fbb_ui .- fbb_u0))
# vi = part .+ (4.5 .* (fbb_ui .- fbb_u0))
normalise!(fbb_tau_suvr, v0, vi)
fbb_tau_conc = map(x -> conc.(x, v0, vi), fbb_tau_suvr)
fbb_tau_inits = [d[:,1] for d in fbb_tau_suvr]

mean_tau_init = mean(fbb_tau_inits)[1:36]
tau_init_idx = findall(x -> x < 0.05, conc.(mean_tau_init, v0[1:36], vi[1:36]))
mean_tau_init[tau_init_idx] .= v0[1:36][tau_init_idx]
CairoMakie.activate!()
scatter(conc.(mean_tau_init, v0[1:36], vi[1:36]))
findall(x -> x > 0, conc.(mean_tau_init, v0[1:36], vi[1:36]))
vol_init = zeros(36)

atn_pkpd = make_atn_pkpd_model(fbb_u0[1:36], fbb_ui[1:36], v0[1:36], part[1:36], Lh, Ld, m, 28)

# ts = range(0, 240, 480)
ts = range(0, 360, 720)

amyloid_production = 0.24 / 12
tau_transport = 0.03 / 12
tau_production = 0.1 / 12
coupling = 3.2258211441306877
atrophy = 0.08 / 12
drug_concentration = 100.
drug_transport = 0.5 / 12
drug_effect = 0.0 / 12
drug_clearance = 0.5 / 12
sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36);mean_fbb_init], 
                (0.0, 360.0), [amyloid_production, tau_transport, tau_production, 
                                        coupling, atrophy, 
                                        drug_transport, drug_effect, 
                                        drug_concentration, drug_clearance]; 
                                        saveat=ts, tol=1e-9)

plot(sol, idxs=1:36)
plot(sol, idxs=37:72)
plot(sol, idxs=73:108)
plot(sol, idxs=109:144)
plot(sol, idxs=145:180)

absol = conc.(Array(sol[1:36,:]), fbb_u0[1:36], fbb_ui[1:36])
tausol = conc.(Array(sol[37:72,:]), v0[1:36], vi[1:36])

begin
    f = Figure()
    ax1 = Axis(f[1,1])
    ax2 = Axis(f[1,2])
    ax3 = Axis(f[1,3])
    for i in 1:36
        lines!(ax1, sol.t, absol[i, :])
        lines!(ax2, sol.t, tausol[i, :])
    end
    scatter!(ax3, tausol[:, 1])
    scatter!(ax3, absol[:, 1])
    f
end

atrsol = Array(sol[73:108,:])
drugsol = Array(sol[109:144,:]) ./ maximum(Array(sol[109:144,:]))

tau_seed = findall(x -> x >= 0.1, tausol)
tau_seed_idx = zeros(36, size(sol,2))
tau_seed_idx[tau_seed] .= 1.0
# heatmap(tau_seed_idx)

ab_seed = findall(x -> x >= 0.88, absol)
ab_seed_idx = zeros(36, size(sol,2))
ab_seed_idx[ab_seed] .= 1.0
# heatmap(ab_seed_idx)

ab_tau_coloc = tau_seed_idx .* ab_seed_idx
# heatmap(ab_tau_coloc)
coloc_t = findfirst(x -> x > 0, sum(ab_tau_coloc, dims=1))
coloc_node = findall(x -> x > 0, ab_tau_coloc[:, coloc_t[2]])
dktnames[coloc_node][1]
sol.t[coloc_t[2]]

# bs = get_braak_regions()
# rbs = [filter(x -> x < 37, b) for b in bs]
# b3 = reduce(vcat, rbs[1:3])

vol_rois = ["entorhinal", "Left-Hippocampus", "Right-Hippocampus", "Left-Amygdala", "Right-Amygdala",
"inferiortemporal", "middletemporal", "inferiorparietal", "precuneus"]
rois = findall(x -> x ∈ vol_rois, get_label.(cortex))

ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);
abcmap = ColorScheme(ab_c);
taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
atrcmap = ColorScheme(atr_c); #ColorSchemes.Reds;

# amyloid_production = 1. / 12
# tau_transport = 0.2 / 12
# tau_production = 0.25 /12
# coupling = 4.5
# atrophy = 0.1 / 12
# drug_concentration = 200.
# drug_transport = 0.5
drug_effect = 0.0
# drug_clearance = 0.1
noab_sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36); mean_fbb_init], 
                (0.0, 360.0), [amyloid_production, tau_transport, tau_production, 
                                        0.0, atrophy, 
                                        drug_transport, drug_effect, 
                                        drug_concentration, drug_clearance]; 
                                        saveat=ts, tol=1e-12)
noab_tau = mean(Array(noab_sol[37:72,end])[rois])
noab_atr= mean(Array(noab_sol[73:108,end])[rois])

absols = Vector{Array{Float64}}()
tausols = Vector{Array{Float64}}()
atrsols = Vector{Array{Float64}}()
drugsols = Vector{Array{Float64}}()
int_ts = collect(0:24:360)
for (i, t) in enumerate(int_ts)
    # atn_pkpd = make_atn_pkpd_model(fbb_ui[1:36] .- fbb_u0[1:36], part[1:36] .- v0[1:36], Lh, Ld, m, t)
    atn_pkpd = make_atn_pkpd_model(fbb_u0[1:36], fbb_ui[1:36], v0[1:36], part[1:36], Lh, Ld, m, t)
    
    # amyloid_production = 1. / 12
    # tau_transport = 0.2 / 12
    # tau_production = 0.06 /12
    # coupling = 4.5
    # atrophy = 0.1 / 12
    drug_concentration = 100.
    drug_transport = 0.5 / 12
    drug_effect = 0.5 / 12
    drug_clearance = 0.5 / 12
    sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36); mean_fbb_init], 
                    (0.0, 360.0), [amyloid_production, tau_transport, tau_production, 
                                            coupling, atrophy, 
                                            drug_transport, drug_effect, 
                                            drug_concentration, drug_clearance]; 
                                            saveat=ts, tol=1e-12)

    push!(absols, conc.(Array(sol[1:36,:]), fbb_u0[1:36], fbb_ui[1:36]))
    push!(tausols, conc.(Array(sol[37:72,:]), v0[1:36], vi[1:36]))
    push!(atrsols, Array(sol[73:108,:]))
    push!(drugsols, Array(sol[109:144,:]))    
end

begin
    GLMakie.activate!()
    cmap = ColorSchemes.Blues
    fig = Figure(size=(1200, 1000), fontsize=15)
    g = fig[1,1] = GridLayout()
    g1 = g[1:4, 1] = GridLayout() 
    G2 = g[1:4, 2] = GridLayout()
    g2 = G2[1, 1] = GridLayout()
    g3 = G2[2, 1]= GridLayout()
    ax1 = Axis(g1[3:4,1], ylabel="Aβ", ytickformat="{:.1f}", ylabelsize=20, xlabelsize=20, xticks=0:24:360)
    hidexdecorations!(ax1, grid=false)
    ylims!(ax1, 0.0, 1.05)
    ax2 = Axis(g1[5:6,1], ylabel="Tau", ytickformat="{:.1f}", xlabel="Time / Months", ylabelsize=20, xlabelsize=20, xticks=0:24:240)
    hidexdecorations!(ax2, grid=false)
    ylims!(ax2, 0.0, 1.05)
    ax3 = Axis(g1[7:8,1], ylabel="Atr", ytickformat="{:.2f}", xlabel="Time / Months", ylabelsize=20, xlabelsize=20, xticks=0:24:240)
    ylims!(ax3, 0.0, 1.05)
    # hidexdecorations!(ax3, grid=false)
    ax4 = Axis(g1[1:2,1], ylabel="Drug\nμg / ml", ytickformat="{:.0f}", xlabel="Time / Months", ylabelsize=20, xlabelsize=20, xticks=0:24:240)
    ylims!(ax4, 0.0, 250.05)
    hidexdecorations!(ax4, grid=false)
   

    ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
    tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
    atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);
    abcmap = ColorScheme(ab_c);
    taucmap = ColorScheme(tau_c); #reverse(ColorSchemes.RdYlBu);
    atrcmap = ColorScheme(atr_c); 

    nodes = get_node_id.(cortex)
    ax_init_1 = Axis3(g2[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax_init_1)
    hidespines!(ax_init_1)
    plot_roi!(nodes, absols[1][:,1], abcmap)
    ax_init_1 = Axis3(g2[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax_init_1)
    hidespines!(ax_init_1)
    plot_roi!(nodes, absols[1][:,1], abcmap)
    cb = Colorbar(g2[2, 1:2], limits = (0.0, 1.0), colormap = abcmap, label="Aβ",
    vertical = false, labelsize=20, flipaxis=false, ticks=collect(0:0.5:1))
    cb.alignmode = Mixed(right = 0)

    ax_init_2 = Axis3(g2[1,3], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax_init_2)
    hidespines!(ax_init_2)
    plot_roi!(nodes, tausols[1][:,1], taucmap)
    ax_init_2 = Axis3(g2[1,4], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    hidedecorations!(ax_init_2)
    hidespines!(ax_init_2)
    plot_roi!(nodes, tausols[1][:,1], taucmap)
    cb = Colorbar(g2[2, 3:4], limits = (0.0, 1.0), colormap = taucmap, label="Tau",
    vertical = false, labelsize=20, flipaxis=false, ticks=collect(0:0.5:1))
    cb.alignmode = Mixed(right = 0)

    ax = Axis(g3[1,1], xlabel="Tau", ylabel="Atr.", xlabelsize=25, ylabelsize=25 )
    # xlims!(ax, 0.4, 1.0)
    # ylims!(ax, 0, 0.7)
    tau_end = [mean(t[rois, end]) for t in tausols]
    atr_end = [mean(t[rois, end]) for t in atrsols]
    for i in eachindex(tau_end)
        linesegments!([0, tau_end[i]], [atr_end[i], atr_end[i]], linestyle=:dash, color=(:grey, 0.75))
        linesegments!([tau_end[i], tau_end[i]], [0, atr_end[i]], linestyle=:dash, color=(:grey, 0.75))
    end
    sc = scatter!(tau_end, atr_end)
    # scatter!(noab_tau, noab_atr)

    ax.alignmode = Mixed(left = 0, right = 0)

    ax = Axis(g3[1,2], xlabel="t0", ylabel="Δ Atr.", xlabelsize=25, ylabelsize=25,xticks=collect(0:60:300))
    ylims!(ax, 0, 0.15)
    tau_end = [mean(t[rois, end]) for t in tausols]
    atr_end = [mean(t[rois, end]) for t in atrsols]
    scatter!(collect(24:24:360), [atr_end[i + 2] - atr_end[i + 1] for i in 0:14], label="Atr")
    scatter!(collect(24:24:360), [tau_end[i + 2] - tau_end[i + 1] for i in 0:14], label="Tau")
    # scatter!(collect(6:6:120), [tau_end[i + 2] - tau_end[i + 1] for i in 0:19])
    vlines!(sol.t[coloc_t[2]], linestyle=:dash, color=:black, linewidth=2.5)
    ax.alignmode = Mixed(left = 0, right = 0)
    axislegend(ax, unique=true, position=:rt,  framevisible=false, fontsize=25, patchsize=(25,25))


    # ax_init_3 = Axis3(g[3][1,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    # hidedecorations!(ax_init_3)
    # hidespines!(ax_init_3)
    # ax_init_3 = Axis3(g[3][1,3], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    # hidedecorations!(ax_init_3)
    # hidespines!(ax_init_3)

    # ax_init_1 = Axis3(g[4][1,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    # hidedecorations!(ax_init_1)
    # hidespines!(ax_init_1)
    # ax_init_1 = Axis3(g[4][1,3], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
    # hidedecorations!(ax_init_1)
    # hidespines!(ax_init_1)
    linestyles = [:solid, :dash, :dashdot, :dashdotdot, :dot]
    labels = ["t0 = 0", "t0 = 48","t0 = 120","t0 = 216","Placebo"]
    solidx = [1, 3, 6, 10, 16]
    for (_absol, _tausol, _atrsol, _drugsol, ls, label) in zip(absols[solidx], tausols[solidx], atrsols[solidx], drugsols[solidx], reverse(linestyles), labels)
        absol = vec(mean(_absol[rois,:], dims=1))
        tausol = vec(mean(_tausol[rois,:], dims=1))
        atrsol = vec(mean(_atrsol[rois,:], dims=1))
        drugsol = vec(mean(_drugsol[rois,:], dims=1)) 
        lines!(ax1, sol.t, vec(absol), linewidth=3, linestyle = (ls), color=get(abcmap, maximum(absol)))
        lines!(ax2, sol.t, vec(tausol), linewidth=3, linestyle = (ls), color=get(taucmap, maximum(tausol)))
        lines!(ax3, sol.t, vec(atrsol), linewidth=3, linestyle = (ls), color=get(atrcmap, maximum(atrsol)/0.5))
        lines!(ax4, sol.t, vec(drugsol), linewidth=3, linestyle = (ls), color=get(taucmap, 0.8) , label = label)
    end
    # le = [LineElement(color = :black, linestyle = ls) for ls in reverse(linestyles)]
    # Le = Legend(g1[5, 1], le, labels, nbanks=5, framevisible=false, tellheight=false)
    # Le.alignmode = Mixed(bottom=0, top = 0)
    axislegend(ax4, unique=true, position=:lt,  orientation = :horizontal, framevisible=false, fontsize=5, nbanks=2, patchsize=(30,10), padding=(0,0,0,-5))

    # gb = f[2,:] = GridLayout()
    # t0s = collect(0:24:360)[solidx]
    # for (t, l) in zip(1:5, ["t0 = $(t0s[1])", "t0 = $(t0s[2])","t0 = $(t0s[3])","t0 = $(t0s[4])","Placebo"])
    #     Label(gb[0,1+t], l, tellwidth=false, fontsize=20)
    #     nodes = get_node_id.(cortex)
    #     ax_init_1 = Axis3(gb[1,1+t], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    #     hidedecorations!(ax_init_1)
    #     hidespines!(ax_init_1)
    #     plot_roi!(nodes, tausols[solidx][t][:,end], taucmap)
    #     ax_init_1 = Axis3(gb[2,1+t], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    #     hidedecorations!(ax_init_1)
    #     hidespines!(ax_init_1)
    #     plot_roi!(nodes, tausols[solidx][t][:,end], taucmap)
    
    #     ax_init_2 = Axis3(gb[3,1+t], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    #     hidedecorations!(ax_init_2)
    #     hidespines!(ax_init_2)
    #     plot_roi!(nodes, atrsols[solidx][t][:,end], atrcmap)
    #     ax_init_2 = Axis3(gb[4,1+t], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
    #     hidedecorations!(ax_init_2)
    #     hidespines!(ax_init_2)
    #     plot_roi!(nodes, atrsols[solidx][t][:,end], atrcmap) 
    # end
    # cb = Colorbar(gb[1:2, 1], limits = (0.0, 1.0), colormap = taucmap, label="Tau",
    # vertical = true, labelsize=20, flipaxis=false, ticks=collect(0:0.5:1))
    # cb = Colorbar(gb[3:4, 1], limits = (0.0, 1.0), colormap = atrcmap, label="Atr.",
    # vertical = true, labelsize=20, flipaxis=false, ticks=collect(0:0.5:1))
    # Label(gb[1:2, 1], "Tau", rotation = pi/2,  justification = :center, tellheight=false, tellwidth=false)
    # Label(gb[3:4, 1], "Atr", rotation = pi/2,  justification = :center, tellheight=false, tellwidth=false)
    # colsize!(gb, 1, 0)
    # cb.alignmode = Mixed(left = 0)

    colsize!(g, 1, 500)
    rowsize!(G2, 1, 150)
    rowsize!(fig.layout, 1, 450)
    colgap!(fig.layout, 100) 
    rowgap!(fig.layout, 25)

    Label(g1[1, 1, TopLeft()], "A", fontsize = 25, font = :bold, padding = (-15, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
    Label(g2[1, 1, TopLeft()], "B", fontsize = 25, font = :bold, padding = (10, 0, 0, 0), halign = :center, tellheight=false, tellwidth=false)
    Label(g3[1, 1, TopLeft()], "C", fontsize = 25, font = :bold, padding = (10, 0, 0, 0), halign = :center, tellheight=false, tellwidth=false)
    # Label(gb[1, 1, TopLeft()], "D", fontsize = 25, font = :bold, padding = (-25, 0, 0, -75), halign = :left, tellheight=false, tellwidth=false)
    
    display(fig)
    
end
save(projectdir("output/plots/pkpd/atn-pkpd-coloc-2.jpeg"), f)

t0s = collect(0:24:240)
int_ts[solidx]
t0s = [24, 72, 120]
int_sols = Vector{ODESolution}()
for (i, t) in enumerate(t0s)
    atn_pkpd = make_scaled_atn_pkpd_model(fbb_ui[1:36] .- fbb_u0[1:36], part[1:36] .- v0[1:36], Lh, Ld, m, t)

    drug_concentration = 100.
    drug_transport = 0.5 / 12
    drug_effect = 0.075 / 12
    drug_clearance = 0.5 / 12
    _sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36)], 
                    (0.0, 360.0), [amyloid_production, tau_transport, tau_production, 
                                            coupling, atrophy, 
                                            drug_transport, drug_effect, 
                                            drug_concentration, drug_clearance]; 
                                            saveat=ts, tol=1e-12)

    push!(int_sols, _sol)   
end

trial_duration = 18
_inits = [sol(t) for t in t0s]
int_inits = [_sol(t) for (_sol,t) in zip(int_sols, t0s)]
int_outcome = [_sol(t + 18) for (_sol,t) in zip(int_sols, t0s)]
placebo_outcome = [sol(t + 18) for t in t0s]

frontal_rois = findall(x -> contains(x, "frontal"), get_label.(cortex)) .+ 36

int_diff = [mean(io[frontal_rois]) - mean(ii[frontal_rois]) for (io, ii) in zip(int_outcome, int_inits)]
placebo_diff = [mean(io[frontal_rois]) - mean(ii[frontal_rois]) for (io, ii) in zip(placebo_outcome, int_inits)]

begin
    CairoMakie.activate!()
    colors = Makie.wong_colors();
    # Figure and Axis
    colors = Makie.wong_colors()
    fig = Figure()
    ax = Axis(fig[1,1], xticks = (1:3, ["left", "middle", "right"]),
            title = "Dodged bars with legend")

    # Plot
    barplot!(reduce(vcat, [[int_diff[i],placebo_diff[i]] for i in 1:3]))

    # Legend
    labels = ["group 1", "group 2", "group 3"]
    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
    title = "Groups"

    Legend(fig[1,2], elements, labels, title)
    fig
end
fig

solidx = [1, 3, 6, 10, 16]
# begin
#     GLMakie.activate!()
#     cmap = ColorSchemes.Blues
#     f = Figure(size=(1000, 1000), fontsize=15)
#     g = f[1,1] = GridLayout()
#     g1 = g[1:4, 1] = GridLayout() 
#     g2 = g[1:2, 2] = GridLayout()
#     g3 = g[3:4, 2]= GridLayout()
#     ax1 = Axis(g1[3:4,1], ylabel="Aβ", ytickformat="{:.1f}", ylabelsize=20, xlabelsize=20)
#     hidexdecorations!(ax1, grid=false)
#     ylims!(ax1, 0.0, 1.05)
#     ax2 = Axis(g1[5:6,1], ylabel="Tau", ytickformat="{:.1f}", xlabel="Time / Months", ylabelsize=20, xlabelsize=20)
#     hidexdecorations!(ax2, grid=false)
#     ylims!(ax2, 0.0, 1.05)
#     ax3 = Axis(g1[7:8,1], ylabel="Atr", ytickformat="{:.2f}", xlabel="Time / Months", ylabelsize=20, xlabelsize=20)
#     ylims!(ax3, 0.0, 0.525)
#     # hidexdecorations!(ax3, grid=false)
#     ax4 = Axis(g1[1:2,1], ylabel="Drug μg / ml", ytickformat="{:.0f}", xlabel="Time / Months", ylabelsize=20, xlabelsize=20)
#     ylims!(ax4, 0.0, 250.05)
#     hidexdecorations!(ax4, grid=false)
#     absols = Vector{Array{Float64}}()
#     tausols = Vector{Array{Float64}}()
#     atrsols = Vector{Array{Float64}}()
#     drugsols = Vector{Array{Float64}}()
#     for (i, t) in enumerate(0:6:120)
#         atn_pkpd = make_scaled_atn_pkpd_model(fbb_ui[1:36] .- fbb_u0[1:36], part[1:36], Lh, Ld, m, t)

#         amyloid_production = 1. / 12
#         tau_transport = 0.2 / 12
#         tau_production = 0.25 /12
#         coupling = 4.5
#         atrophy = 0.1 / 12
#         drug_concentration = 200.
#         drug_transport = 0.5
#         drug_effect = 0.0
#         drug_clearance = 0.1
#         sol = simulate(atn_pkpd, [mean_fbb_init; mean_tau_init; vol_init; zeros(36)], 
#                         (0.0, 120.0), [amyloid_production, tau_transport, tau_production, 
#                                                 coupling, atrophy, 
#                                                 drug_transport, drug_effect, 
#                                                 drug_concentration, drug_clearance]; 
#                                                 saveat=ts, tol=1e-12)

#         push!(absols, Array(sol[1:36,:]))
#         push!(tausols, Array(sol[37:72,:]))
#         push!(atrsols, Array(sol[73:108,:]))
#         push!(drugsols, Array(sol[109:end,:]))    
#     end

#     nodes = get_node_id.(cortex)
#     ax_init_1 = Axis3(g2[1,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#     hidedecorations!(ax_init_1)
#     hidespines!(ax_init_1)
#     plot_roi!(nodes, absols[1][:,1], abcmap)
#     ax_init_1 = Axis3(g2[1,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#     hidedecorations!(ax_init_1)
#     hidespines!(ax_init_1)
#     plot_roi!(nodes, absols[1][:,1], abcmap)
#     cb = Colorbar(g2[1, 3], limits = (0.0, 1.0), colormap = abcmap, label="Aβ",
#     vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.5:1))
#     cb.alignmode = Mixed(right = 0)


#     ax_init_2 = Axis3(g2[2,1], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#     hidedecorations!(ax_init_2)
#     hidespines!(ax_init_2)
#     plot_roi!(nodes, tausols[1][:,1], taucmap)
#     ax_init_2 = Axis3(g2[2,2], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#     hidedecorations!(ax_init_2)
#     hidespines!(ax_init_2)
#     plot_roi!(nodes, tausols[1][:,1], taucmap)
#     cb = Colorbar(g2[2, 3], limits = (0.0, 1.0), colormap = taucmap, label="Tau",
#     vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.5:1))
#     cb.alignmode = Mixed(right = 0)

#     ax = Axis(g3[1,1], xlabel="Tau", ylabel="Neurodegeneration")
#     xlims!(ax, 0, 1.0)
#     ylims!(ax, 0, 0.5)
#     tau_end = [mean(t[rois, end]) for t in tausols]
#     atr_end = [mean(t[rois, end]) for t in atrsols]
#     sc = scatter!(tau_end, atr_end)
#     scatter!(noab_tau, noab_atr)
#     for i in eachindex(tau_end)
#         linesegments!([0, tau_end[i]], [atr_end[i], atr_end[i]], linestyle=:dash, color=(:grey, 0.75))
#         linesegments!([tau_end[i], tau_end[i]], [0, atr_end[i]], linestyle=:dash, color=(:grey, 0.75))
#     end
#     ax.alignmode = Mixed(left = 0, right = 0)


#     # ax_init_3 = Axis3(g[3][1,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#     # hidedecorations!(ax_init_3)
#     # hidespines!(ax_init_3)
#     # ax_init_3 = Axis3(g[3][1,3], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#     # hidedecorations!(ax_init_3)
#     # hidespines!(ax_init_3)

#     # ax_init_1 = Axis3(g[4][1,2], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#     # hidedecorations!(ax_init_1)
#     # hidespines!(ax_init_1)
#     # ax_init_1 = Axis3(g[4][1,3], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(1.0,1.0,1.0,1.0))
#     # hidedecorations!(ax_init_1)
#     # hidespines!(ax_init_1)
#     linestyles = [:solid, :dash, :dashdot, :dashdotdot, :dot]
#     labels = ["t0 = 0", "t0 = 6","t0 = 12","t0 = 36","Placebo"]
#     solidx = [1, 3, 5, 10, 21]
#     for (_absol, _tausol, _atrsol, _drugsol, ls, label) in zip(absols[solidx], tausols[solidx], atrsols[solidx], drugsols[solidx], reverse(linestyles), labels)
#         absol = vec(mean(_absol[rois,:], dims=1))
#         tausol = vec(mean(_tausol[rois,:], dims=1))
#         atrsol = vec(mean(_atrsol[rois,:], dims=1))
#         drugsol = vec(mean(_drugsol[rois,:], dims=1)) 
#         lines!(ax1, sol.t, vec(absol), linewidth=3, linestyle = (ls, :dense), color=get(abcmap, maximum(absol)))
#         lines!(ax2, sol.t, vec(tausol), linewidth=3, linestyle = (ls, :dense), color=get(taucmap, maximum(tausol)/0.4))
#         lines!(ax3, sol.t, vec(atrsol), linewidth=3, linestyle = (ls, :dense), color=get(atrcmap, maximum(atrsol)/0.15))
#         lines!(ax4, sol.t, vec(drugsol), linewidth=3, linestyle = (ls, :dense), color=get(taucmap, 0.8) , label = label)
#     end
#     # le = [LineElement(color = :black, linestyle = ls) for ls in reverse(linestyles)]
#     # Le = Legend(g1[5, 1], le, labels, nbanks=5, framevisible=false, tellheight=false)
#     # Le.alignmode = Mixed(bottom=0, top = 0)
#     axislegend(ax4, unique=true, position=:lt,  orientation = :horizontal, framevisible=false, fontsize=10)
#     gb = f[2,:] = GridLayout()

#     for (t, l) in zip(1:5, ["t0 = 0", "t0 = 6","t0 = 12","t0 = 36","Placebo"])
#         Label(gb[0,1+t], l, tellwidth=false, fontsize=20)
#         nodes = get_node_id.(cortex)
#         ax_init_1 = Axis3(gb[1,1+t], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
#         hidedecorations!(ax_init_1)
#         hidespines!(ax_init_1)
#         plot_roi!(nodes, tausols[solidx][t][:,end], taucmap)
#         ax_init_1 = Axis3(gb[2,1+t], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
#         hidedecorations!(ax_init_1)
#         hidespines!(ax_init_1)
#         plot_roi!(nodes, tausols[solidx][t][:,end], taucmap)
    
#         ax_init_2 = Axis3(gb[3,1+t], aspect = :data, azimuth = 0.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
#         hidedecorations!(ax_init_2)
#         hidespines!(ax_init_2)
#         plot_roi!(nodes, atrsols[solidx][t][:,end], atrcmap)
#         ax_init_2 = Axis3(gb[4,1+t], aspect = :data, azimuth = 1.0pi, elevation=0.0pi, protrusions=(0.0,0.0,0.0,0.0))
#         hidedecorations!(ax_init_2)
#         hidespines!(ax_init_2)
#         plot_roi!(nodes, atrsols[solidx][t][:,end], atrcmap) 
#     end
#     cb = Colorbar(gb[1:2, 1], limits = (0.0, 1.0), colormap = taucmap, label="Tau",
#     vertical = true, labelsize=20, flipaxis=false, ticks=collect(0:0.5:1))
#     cb = Colorbar(gb[3:4, 1], limits = (0.0, 1.0), colormap = atrcmap, label="Atr.",
#     vertical = true, labelsize=20, flipaxis=false, ticks=collect(0:0.5:1))
#     # Label(gb[1:2, 1], "Tau", rotation = pi/2,  justification = :center, tellheight=false, tellwidth=false)
#     # Label(gb[3:4, 1], "Atr", rotation = pi/2,  justification = :center, tellheight=false, tellwidth=false)
#     # colsize!(gb, 1, 0)
#     # cb.alignmode = Mixed(left = 0)

#     colsize!(g, 1, 500)
#     rowsize!(f.layout, 1, 450)
#     colgap!(f.layout, 100) 
#     rowgap!(f.layout, 25)

#     Label(g1[1, 1, TopLeft()], "A", fontsize = 26, font = :bold, padding = (-15, 0, 0, 0), halign = :left, tellheight=false, tellwidth=false)
#     Label(g2[1, 1, TopLeft()], "B", fontsize = 26, font = :bold, padding = (10, 0, 0, 0), halign = :center, tellheight=false, tellwidth=false)
#     Label(g3[1, 1, TopLeft()], "C", fontsize = 26, font = :bold, padding = (10, 0, 0, 0), halign = :center, tellheight=false, tellwidth=false)
#     Label(gb[1, 1, TopLeft()], "D", fontsize = 26, font = :bold, padding = (-25, 0, 0, -75), halign = :left, tellheight=false, tellwidth=false)
    
#     display(f)
    
# end
# # save(projectdir("output/plots/pkpd/atn-pkpd-coloc.jpeg"), f)

# scatter(collect(6:6:120), [atr_end[i + 2] - atr_end[i + 1] for i in 0:19])

# scatter([tau_end[i+2] - tau_end[i + 1] for i in 0:19], [atr_end[i + 2] - atr_end[i + 1] for i in 0:19])

# scatter(collect(0:6:120), [atr_end[i + 2] - atr_end[i + 1] for i in 0:20])

# scatter([tau_end[i+2] - tau_end[i + 1] for i in 0:19], [atr_end[i + 2] - atr_end[i + 1] for i in 0:19])

# for st in ([0,0], [1,0], [1,1])
#     # Amyloid data 
#     _ab_data_df =  CSV.read(datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv"), DataFrame)
#     _tau_data_df = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame) 

#     tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
#     tau_pos_df = filter(x ->  x.MTL_Status == st[1] && x.NEO_Status == st[2], tau_data_df);
#     tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=1)
#     IDS = unique(tau_pos_df.RID)
#     # --------------------------------------------------------------------------------
#     # Load fbb data
#     # --------------------------------------------------------------------------------
#     tracer="FBB"
    
#     fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
#     fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer && x.RID ∈ IDS, _ab_data_df)
#     fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF")

#     fbb_suvr = calc_suvr.(fbb_data)
#     normalise!(fbb_suvr, fbb_u0, fbb_ui)
#     fbb_conc = map(x -> conc.(x, fbb_u0, fbb_ui), fbb_suvr)
#     fbb_inits = [d[:,1] for d in fbb_conc]
#     mean_fbb_inits = mean(fbb_inits)[1:36]
#     println("ab concentration = $(mean_fbb_inits[29])")

#     fbb_tau_suvr = calc_suvr.(tau_data)
#     normalise!(fbb_tau_suvr, v0, vi)
#     fbb_tau_conc = map(x -> conc.(x, v0, vi), fbb_tau_suvr)
#     fbb_tau_inits = [d[:,1] for d in fbb_tau_conc]
#     mean_tau_inits = mean(fbb_tau_inits)[1:36]
#     println("tau concentration = $(mean_tau_inits[29])")

#     vol_inits = zeros(36)

#     atn_pkpd = make_scaled_atn_pkpd_model(fbb_ui[1:36] .- fbb_u0[1:36], part[1:36], Lh, Ld, m)

#     ts = range(0, 60, 600)

#     amyloid_production = 0.2
#     tau_transport = 0.04
#     tau_production = 0.05
#     coupling = 4.5
#     atrophy = 0.05
#     drug_concentration = 10.
#     drug_transport = 0.1
#     drug_effect = 0.05
#     drug_clearance = 0.1

#     sol = simulate(atn_pkpd, [mean_fbb_inits; mean_tau_inits; vol_inits; zeros(36)], 
#                     (0.0, 60.0), [amyloid_production, tau_transport, tau_production, 
#                                             coupling, atrophy, 
#                                             drug_transport, drug_effect, 
#                                             drug_concentration, drug_clearance]; 
#                                             saveat=ts, tol=1e-9)

#     absol = Array(sol[1:36,:])
#     tausol = Array(sol[37:72,:])
#     atrsol = Array(sol[73:108,:])
#     drugsol = Array(sol[109:end,:]) ./ maximum(Array(sol[109:end,:]))

#     begin
#         CairoMakie.activate!()
#         cmap = Makie.wong_colors()
#         f = Figure()
#         ax1 = Axis(f[1,1], ylabel="Amyloid Conc", ytickformat="{:.0f}")
#         hidexdecorations!(ax1, grid=false)
#         ylims!(ax1, 0.0, 1.05)
#         ax2 = Axis(f[2,1], ylabel="Tau Conc", ytickformat="{:.0f}", xlabel="Time / Months")
#         hidexdecorations!(ax2, grid=false)
#         ylims!(ax2, 0.0, 1.05)
#         ax3 = Axis(f[3,1], ylabel="Atr", ytickformat="{:.0f}", xlabel="Time / Months")
#         ylims!(ax3, 0.0, 1.05)
#         hidexdecorations!(ax3, grid=false)
#         ax4 = Axis(f[4,1], ylabel="Drug Conc", ytickformat="{:.0f}", xlabel="Time / Months")
#         ylims!(ax4, 0.0, 1.05)
#         for i in 1:36
#             lines!(ax1, sol.t, absol[i,:], color=cmap[1])
#             lines!(ax2, sol.t, tausol[i,:], color=cmap[1])
#             lines!(ax3, sol.t, atrsol[i,:], color=cmap[1])
#             lines!(ax4, sol.t, drugsol[i,:], color=cmap[1])
#         end
#         display(f)
#     end
# end