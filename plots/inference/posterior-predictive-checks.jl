using ATNModelling.SimulationUtils: make_prob, make_atn_model, make_atn_fixed_model,
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names,
                                    get_braak_regions
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise
using ATNModelling.InferenceModels: fit_model, ensemble_atn, serial_atn, fit_serial_atn

using Connectomes: laplacian_matrix, get_label, get_node_id
using ADNIDatasets: ADNIDataset, get_id, get_dates, get_initial_conditions, 
                    calc_suvr, get_vol, get_times, data_dashboard
using DrWatson: projectdir, datadir
using CSV, DataFrames
using SciMLBase: successful_retcode
using DifferentialEquations, Turing, LinearAlgebra
using Random
using Serialization
# --------------------------------------------------------------------------------
# Load parameters
# --------------------------------------------------------------------------------
u0, ui = load_ab_params()
ui_diff = ui .- u0
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c)

# --------------------------------------------------------------------------------
# Loading data and aligning
# --------------------------------------------------------------------------------
cortex = get_parcellation() |> get_cortex
dktnames = get_dkt_names(cortex)

# Amyloid data 
_ab_data_df =  CSV.read(datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame) 

ab_data_df = filter(x -> x.qc_flag==2 && x.TRACER == "FBP", _ab_data_df);
ab_data = ADNIDataset(ab_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

# Tau data 
tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=3)

ab, tau = align_data(ab_data, tau_data; min_tau_scans=3)

ab_times = get_times.(ab)
tau_times = get_times.(tau)
ts = [sort(unique([a; t])) for (a, t) in zip(ab_times, tau_times)]

ab_tidx = get_time_idx(ab_times, ts)
tau_tidx = get_time_idx(tau_times, ts)

@assert allequal([allequal(ab_times[i] .== ts[i][ab_tidx[i]]) for i in 1:22])
@assert allequal([allequal(tau_times[i] .== ts[i][tau_tidx[i]]) for i in 1:22])

ab_suvr = calc_suvr.(ab)
normalise!(ab_suvr, u0, ui)
ab_inits = [d[:,1] for d in ab_suvr]

tau_suvr = calc_suvr.(tau)
normalise!(tau_suvr, v0)
tau_inits = [d[:,1] for d in tau_suvr]

tau_pos_vol = get_vol.(tau)
total_vol_norm = [tp ./ sum(tp, dims=1) for tp in tau_pos_vol]
vols = [clamp.(1 .- (vol ./ vol[:,1]), 0, 1) for vol in total_vol_norm]
vol_inits = [vol[:,1] for vol in vols]

atn_model = make_atn_model(u0, ui, v0, part, L)
prob = make_prob(atn_model, 
          [ab_inits[1]; tau_inits[1]; vol_inits[1]], 
          (0.0,7.5), [1.0,1.0,1.0,3.5,1.0])
sol = solve(prob, Tsit5())

using Plots
Plots.plot(sol, idxs=73:144, labels=false)

inits = [[ab; tau; vol] for (ab, tau, vol) in zip(ab_inits, tau_inits, vol_inits)]
n_subjects = length(ab)

pst = deserialize(projectdir("output/chains/population-atn/pst-samples-lognormal-2-1x1000.jls"));
summarize(pst)
meanpst = mean(pst)

sub = 22
prob = make_prob(atn_model, 
          [ab_inits[sub]; tau_inits[sub]; vol_inits[sub]], 
          (0.0,50.5), 
            [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
            meanpst[Symbol("α_t[$sub]"), :mean], meanpst[Symbol("β"), :mean], 
            meanpst[Symbol("η[$sub]"), :mean]])
sol = solve(prob, Tsit5())

using Plots
Plots.plot(sol, idxs=73:144, labels=false)
Plots.plot(sol, idxs=1:72, labels=false)

[display(scatter(vec(pst["α_t[$i]"]), vec(pst[:β]))) for i in 1:22]

function get_diff(d)
    d[:,end] .- d[:,1]
end

using GLMakie, Colors, ColorSchemes, Connectomes
data_dashboard(tau, v0, vi; show_mtl_threshold=true)

using CairoMakie; CairoMakie.activate!()
begin
    ab_sols = Vector{Array{Float64}}()
    tau_sols = Vector{Array{Float64}}()
    atr_sols = Vector{Array{Float64}}()
    # pst = deserialize(projectdir("analysis/output/chains/testatnpst-beta-u-conc-normal.jls"));
    meanpst = mean(pst)
    for sub in 1:22
    p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
    meanpst[Symbol("α_t[$sub]"), :mean], meanpst[Symbol("β"), :mean], 
    meanpst[Symbol("η[$sub]"), :mean]]

    pstprob = ODEProblem(atn_model, inits[sub], (0.0, 10.0), p)
    pstsol = solve(pstprob, Tsit5())
    push!(ab_sols,pstsol(ab_times[sub])[1:72,:])
    push!(tau_sols,pstsol(tau_times[sub])[73:144,:])
    push!(atr_sols,pstsol(tau_times[sub])[145:216,:])
    end 
    
    for sub in 1:22
    f = Figure(size=(1500, 600))

    ax1 = CairoMakie.Axis(f[1,1])
    CairoMakie.xlims!(ax1, -0.5, 0.5)
    CairoMakie.ylims!(ax1, -0.5, 0.5)
    
    ax2 = CairoMakie.Axis(f[1,2])
    CairoMakie.xlims!(ax2, -1.0, 1.0)
    CairoMakie.ylims!(ax2, -1.0, 1.0)

    ax3 = CairoMakie.Axis(f[1,3])
    CairoMakie.xlims!(ax3, -0., 0.5)
    CairoMakie.ylims!(ax3, -0., 0.5)

    CairoMakie.scatter!(ax1, get_diff(ab_suvr[sub]), get_diff(ab_sols[sub]))
    CairoMakie.scatter!(ax2, get_diff(tau_suvr[sub]), get_diff(tau_sols[sub]))
    CairoMakie.scatter!(ax3, get_diff(vols[sub]), get_diff(atr_sols[sub]))
    display(f)
    end
end

braak_regions = get_braak_regions()
bs = [findall(x -> get_node_id(x) ∈ br, cortex) for br in braak_regions]

begin
    cmap = Makie.wong_colors();

    f = Figure(size=(1500, 500))
    titlesize = 40
    xlabelsize = 25 
    ylabelsize = 25
    xticklabelsize = 20 
    yticklabelsize = 20

    start = -0.02
    stop = 0.08
    border = 0.005
    ax4 =CairoMakie.Axis(f[1,1],  
            xlabel="Δ SUVR", 
            ylabel="Δ Prediction", 
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
            xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
            xticks=start:0.02:stop, yticks=start:0.02:stop, 
            xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
            xtickformat = "{:.3f}", ytickformat = "{:.3f}")

    CairoMakie.xlims!(ax4, start, stop + border)
    CairoMakie.ylims!(ax4, start, stop + border)
    lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
    start = -0.0
    stop = 0.3
    border = 0.01
    ax5 =CairoMakie.Axis(f[1,2],  
            xlabel="Δ SUVR", 
            ylabel="Δ Prediction", 
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
            xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
            xticks=start:0.05:stop, yticks=start:0.05:stop, 
            xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
            xtickformat = "{:.2f}", ytickformat = "{:.2f}")
    hideydecorations!(ax5, grid=false, ticks=false, ticklabels=false)
    CairoMakie.xlims!(ax5, start, stop + border)
    CairoMakie.ylims!(ax5, start, stop + border)
    lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

    start = -0.0
    stop = 0.15
    border = 0.01
    ax6= CairoMakie.Axis(f[1,3],  
            xlabel="Δ Atr.", 
            ylabel="Δ Prediction", 
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
            xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
            xticks=start:0.05:stop, yticks=start:0.05:stop, 
            xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
            xtickformat = "{:.2f}", ytickformat = "{:.2f}")
    hideydecorations!(ax6, grid=false, ticks=false, ticklabels=false)
    CairoMakie.xlims!(ax6, start, stop + border)
    CairoMakie.ylims!(ax6, start, stop + border)
    lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
    ab_sols = Vector{Array{Float64}}()
    tau_sols = Vector{Array{Float64}}()
    atr_sols = Vector{Array{Float64}}()
    meanpst = mean(pst)

    for sub in 1:22
        p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
        meanpst[Symbol("α_t[$sub]"), :mean], 1.0, #meanpst[Symbol("β"), :mean], 
        meanpst[Symbol("η[$sub]"), :mean]]
        
        pstprob = remake(prob, u0=inits[sub], p=p)
        pstsol = solve(pstprob, Tsit5())
        push!(ab_sols,pstsol(ab_times[sub])[1:72,:])
        push!(tau_sols,pstsol(tau_times[sub])[73:144,:])
        push!(atr_sols,pstsol(tau_times[sub])[145:216,:])
    end
    mean_ab_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in ab_sols]), dims=2))
    mean_tau_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in tau_sols]), dims=2))
    mean_atr_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in atr_sols]), dims=2))
    
    mean_ab =  vec(mean(reduce(hcat, [d[:, end] for d in ab_suvr]), dims=2))
    mean_tau =  vec(mean(reduce(hcat, [d[:, end] for d in tau_suvr]), dims=2))
    mean_atr =  vec(mean(reduce(hcat, [d[:, end] for d in vols]), dims=2))
    
    mean_ab_sol_diff = vec(mean(reduce(hcat, get_diff.(ab_sols)), dims=2))
    mean_tau_sol_diff = vec(mean(reduce(hcat, get_diff.(tau_sols)), dims=2))
    mean_atr_sol_diff = vec(mean(reduce(hcat, get_diff.(atr_sols)), dims=2))

    mean_ab_diff = vec(mean(reduce(hcat, get_diff.(ab_suvr)), dims=2))
    mean_tau_diff = vec(mean(reduce(hcat, get_diff.(tau_suvr)), dims=2))
    mean_atr_diff = vec(mean(reduce(hcat, get_diff.(vols)), dims=2))

    labels = ["Braak 1", "Braak 2/3", "Braak 4", "Braak 5", "Braak 6"]
    for (i, rois) in enumerate(bs)
    CairoMakie.scatter!(ax4, mean_ab_diff[rois], mean_ab_sol_diff[rois], color=cmap[i], markersize=20 , label=labels[i])
    CairoMakie.scatter!(ax5, mean_tau_diff[rois], mean_tau_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
    CairoMakie.scatter!(ax6, mean_atr_diff[rois], mean_atr_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
    end
    Legend(f[2,:], ax6, framevisible = false, unique=true, labelsize=35, nbanks=5, tellheight=true)

    f
end


pst = deserialize(projectdir("output/chains/population-atn/pst-samples-lognormal-fixed-2-1x1000.jls"));
using LsqFit
linearmodel(x, p) = p[1] .+ p[2] .* x
fitted_model = curve_fit(linearmodel, ui .- u0, vi, [1.0, 1.0])
println("params = $(fitted_model.param)")
κ, β = fitted_model.param

function get_diff(d)
    d[:,end] .- d[:,1]
end
atn_model = make_atn_fixed_model(u0, ui, v0, L)

using CairoMakie
begin    
    ab_sols = Vector{Array{Float64}}()
    tau_sols = Vector{Array{Float64}}()
    atr_sols = Vector{Array{Float64}}()
    # pst = deserialize(projectdir("analysis/output/chains/testatnpst-beta-u-conc-normal.jls"));
    meanpst = mean(pst)
    for sub in 1:22
    p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
    meanpst[Symbol("α_t[$sub]"), :mean], κ, β, 
    meanpst[Symbol("η[$sub]"), :mean]]
    pstprob = ODEProblem(atn_model, inits[sub], (0.0, 10.0), p)
    pstsol = solve(pstprob, Tsit5())
    push!(ab_sols,pstsol(ab_times[sub])[1:72,:])
    push!(tau_sols,pstsol(tau_times[sub])[73:144,:])
    push!(atr_sols,pstsol(tau_times[sub])[145:216,:])
    end 
    
    for sub in 1:22
    f = Figure(size=(1500, 600))

    ax1 = CairoMakie.Axis(f[1,1])
    CairoMakie.xlims!(ax1, -0.5, 0.5)
    CairoMakie.ylims!(ax1, -0.5, 0.5)
    
    ax2 = CairoMakie.Axis(f[1,2])
    CairoMakie.xlims!(ax2, -1.0, 1.0)
    CairoMakie.ylims!(ax2, -1.0, 1.0)

    ax3 = CairoMakie.Axis(f[1,3])
    CairoMakie.xlims!(ax3, -0., 0.5)
    CairoMakie.ylims!(ax3, -0., 0.5)

    CairoMakie.scatter!(ax1, get_diff(ab_suvr[sub]), get_diff(ab_sols[sub]))
    CairoMakie.scatter!(ax2, get_diff(tau_suvr[sub]), get_diff(tau_sols[sub]))
    CairoMakie.scatter!(ax3, get_diff(vols[sub]), get_diff(atr_sols[sub]))
    display(f)
    end
end

braak_regions = get_braak_regions()
bs = [findall(x -> get_node_id(x) ∈ br, cortex) for br in braak_regions]

begin
    cmap = Makie.wong_colors();

    f = Figure(size=(1500, 500))
    titlesize = 40
    xlabelsize = 25 
    ylabelsize = 25
    xticklabelsize = 20 
    yticklabelsize = 20

    start = -0.02
    stop = 0.08
    border = 0.005
    ax4 =CairoMakie.Axis(f[1,1],  
            xlabel="Δ SUVR", 
            ylabel="Δ Prediction", 
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
            xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
            xticks=start:0.02:stop, yticks=start:0.02:stop, 
            xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
            xtickformat = "{:.3f}", ytickformat = "{:.3f}")

    CairoMakie.xlims!(ax4, start, stop + border)
    CairoMakie.ylims!(ax4, start, stop + border)
    lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
    start = -0.0
    stop = 0.3
    border = 0.01
    ax5 =CairoMakie.Axis(f[1,2],  
            xlabel="Δ SUVR", 
            ylabel="Δ Prediction", 
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
            xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
            xticks=start:0.05:stop, yticks=start:0.05:stop, 
            xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
            xtickformat = "{:.2f}", ytickformat = "{:.2f}")
    hideydecorations!(ax5, grid=false, ticks=false, ticklabels=false)
    CairoMakie.xlims!(ax5, start, stop + border)
    CairoMakie.ylims!(ax5, start, stop + border)
    lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

    start = -0.0
    stop = 0.15
    border = 0.01
    ax6= CairoMakie.Axis(f[1,3],  
            xlabel="Δ Atr.", 
            ylabel="Δ Prediction", 
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
            xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
            xticks=start:0.05:stop, yticks=start:0.05:stop, 
            xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
            xtickformat = "{:.2f}", ytickformat = "{:.2f}")
    hideydecorations!(ax6, grid=false, ticks=false, ticklabels=false)
    CairoMakie.xlims!(ax6, start, stop + border)
    CairoMakie.ylims!(ax6, start, stop + border)
    lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
    ab_sols = Vector{Array{Float64}}()
    tau_sols = Vector{Array{Float64}}()
    atr_sols = Vector{Array{Float64}}()
    meanpst = mean(pst)

    for sub in 1:22
        p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
        meanpst[Symbol("α_t[$sub]"), :mean],  κ, β, 
        meanpst[Symbol("η[$sub]"), :mean]]
        
        pstprob = ODEProblem(atn_model, inits[sub], (0.0, 10.0), p)
        pstsol = solve(pstprob, Tsit5())
        push!(ab_sols,pstsol(ab_times[sub])[1:72,:])
        push!(tau_sols,pstsol(tau_times[sub])[73:144,:])
        push!(atr_sols,pstsol(tau_times[sub])[145:216,:])
    end
    mean_ab_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in ab_sols]), dims=2))
    mean_tau_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in tau_sols]), dims=2))
    mean_atr_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in atr_sols]), dims=2))
    
    mean_ab =  vec(mean(reduce(hcat, [d[:, end] for d in ab_suvr]), dims=2))
    mean_tau =  vec(mean(reduce(hcat, [d[:, end] for d in tau_suvr]), dims=2))
    mean_atr =  vec(mean(reduce(hcat, [d[:, end] for d in vols]), dims=2))
    
    mean_ab_sol_diff = vec(mean(reduce(hcat, get_diff.(ab_sols)), dims=2))
    mean_tau_sol_diff = vec(mean(reduce(hcat, get_diff.(tau_sols)), dims=2))
    mean_atr_sol_diff = vec(mean(reduce(hcat, get_diff.(atr_sols)), dims=2))

    mean_ab_diff = vec(mean(reduce(hcat, get_diff.(ab_suvr)), dims=2))
    mean_tau_diff = vec(mean(reduce(hcat, get_diff.(tau_suvr)), dims=2))
    mean_atr_diff = vec(mean(reduce(hcat, get_diff.(vols)), dims=2))

    labels = ["Braak 1", "Braak 2/3", "Braak 4", "Braak 5", "Braak 6"]
    for (i, rois) in enumerate(bs)
    CairoMakie.scatter!(ax4, mean_ab_diff[rois], mean_ab_sol_diff[rois], color=cmap[i], markersize=20 , label=labels[i])
    CairoMakie.scatter!(ax5, mean_tau_diff[rois], mean_tau_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
    CairoMakie.scatter!(ax6, mean_atr_diff[rois], mean_atr_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
    end
    Legend(f[2,:], ax6, framevisible = false, unique=true, labelsize=35, nbanks=5, tellheight=true)

    f
end
