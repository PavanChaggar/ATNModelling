using ATNModelling.SimulationUtils: load_ab_params, load_tau_params,
                                    make_atn_model, make_prob, simulate, resimulate,
                                    generate_data
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex
using ATNModelling.InferenceModels: atn_inference, fit_model
using Connectomes: laplacian_matrix, get_label, get_hemisphere, get_node_id,
                   plot_roi!
using FileIO
using DrWatson: projectdir
using Serialization

# --------------------------------------------------------------------------------
# Simulation set-up
# --------------------------------------------------------------------------------
u0, ui = load_ab_params()
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2)
L = laplacian_matrix(c)


α_a, ρ_t, α_t, β, η = 0.75, 0.015, 0.5, 3.75, 0.1
params = [α_a, ρ_t, α_t, β, η]

ab_inits = copy(u0)
tau_inits = copy(v0)
atr_inits = zeros(72)

ab_inits .+= 0.2 .* (ui .- u0)

tau_seed_regions = ["entorhinal" ]#,"Left-Amygdala", "Right-Amygdala", "Left-Hippocampus", "Right-Hippocampus"]
tau_seed_idx = findall(x -> get_label(x) ∈ tau_seed_regions, c.parc)
tau_inits[tau_seed_idx] .+= 0.2 .* ((part[tau_seed_idx] .+ β .* (ui[tau_seed_idx] - u0[tau_seed_idx])) .- v0[tau_seed_idx])

inits = [ab_inits; tau_inits; atr_inits]

tspan = (0.0,30.0)

func = make_atn_model(u0, ui, v0, part, L)

ts = LinRange(0.0, 20, 5)
sol_ts = simulate(func, inits, tspan, params; saveat=0.1)
sol_ts = simulate(func, inits, tspan, params; saveat=ts)
# --------------------------------------------------------------------------------
# Simulation visualisatin
# --------------------------------------------------------------------------------
using GLMakie
using Colors, ColorSchemes

_vi = part .+ (3.75 .* (ui .- u0))
cortex = get_parcellation() |> get_cortex
right_cortical_nodes = filter(x -> get_hemisphere(x) == "right", cortex)
nodes = get_node_id.(right_cortical_nodes)

ab_c = sequential_palette(125, s = 0.75, c = 0.9, w =0., b = 0.9);
tau_c = sequential_palette(250, s = 0.9, c = 0.9, w =0.25, b = 0.5);
atr_c = sequential_palette(15, s = 0.9, c = 0.9, w =0.25, b = 0.5);

begin
    f = Figure(size=(1250, 800))

    g1 = f[1, 1:7] = GridLayout()
    g2 = f[2, 1:7] = GridLayout()
    g3 = f[3, 1:7] = GridLayout()

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
    Colorbar(g1[1:2, 8], limits = (minimum(u0), 1.3), colormap = abcmap,
            vertical = true, labelsize=20, flipaxis=true, ticks=collect(0.5:0.25:1.5),
            ticksize=18, ticklabelsize=20, labelpadding=10)
    Colorbar(g2[1:2, 8], limits = (minimum(v0), 3.5), colormap = taucmap,
            vertical = true, labelsize=20, flipaxis=true, ticks=collect(1.0:0.5:4.0),
            ticksize=18, ticklabelsize=20, labelpadding=10)
    Colorbar(g3[1:2, 8], limits = (0, 1), colormap = atrcmap,
            vertical = true, labelsize=20, flipaxis=true, ticks=collect(0:0.2:1.0),
            ticksize=18, ticklabelsize=20, labelpadding=10)

    ax = GLMakie.Axis(g1[1:2,1:2],
            xautolimitmargin = (0, 0), xgridcolor = (:grey, 0.5), xgridwidth = 2,
            xticklabelsize = 25, xticks = ts, xticksize=10,
            xlabel="Time / years", xlabelsize = 25,
            yautolimitmargin = (0, 0), ygridcolor = (:grey, 0.5), ygridwidth = 2,
            yticklabelsize = 25, yticksize=10,
            ylabel="SUVR", ylabelsize = 25, yticks=collect(0.5:0.25:1.5)
    )
    hidexdecorations!(ax, ticks=false, grid=false)
    GLMakie.ylims!(ax, 0.5-0.05, 1.3+0.05)
    GLMakie.xlims!(ax, 0, 20)
    # vlines!(ax, sol.t[arrival_order[1][2]], color=:black, linewidth=2.0, linestyle=:dash)
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
    GLMakie.ylims!(ax, minimum(v0)-0.15, 3.5+0.15)
    GLMakie.xlims!(ax, 0, 20)
    # vlines!(ax, sol.t[arrival_order[1][2]], color=:black, linewidth=2.0, linestyle=:dash)
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
    GLMakie.ylims!(ax, 0-0.05, 1.0+0.05)
    GLMakie.xlims!(ax, 0, 20)
    # vlines!(ax, sol.t[arrival_order[1][2]], color=:black, linewidth=2.0, linestyle=:dash)
    for i in 1:36
    lines!(sol.t, sol[i+144, :], linewidth=2.0, color=alphacolor(atr_col, 0.5))
    end
    Label(f[1,0], "A", fontsize=30, tellheight=false)
    Label(f[2,0], "T", fontsize=30, tellheight=false)
    Label(f[3,0], "N", fontsize=30, tellheight=false)
    f
end
save(projectdir("output/plots/simulation/atn-simulation.jpeg"), f)