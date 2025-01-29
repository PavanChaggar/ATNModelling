using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, make_atn_fixed_model,
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params, conc
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
tracer = "FBB"
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c)
cortex = get_parcellation() |> get_cortex 
dktnames =  get_dkt_names(cortex)

_ab_data_df =  CSV.read(datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=3)

# --------------------------------------------------------------------------------
# Loading FBB data and aligning
# --------------------------------------------------------------------------------
tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

fbb, fbb_tau = align_data(fbb_data, tau_data; min_tau_scans=3)

fbb_times = get_times.(fbb)
fbb_tau_times = get_times.(fbb_tau)
fbb_ts = [sort(unique([a; t])) for (a, t) in zip(fbb_times, fbb_tau_times)]

fbb_ab_tidx = get_time_idx(fbb_times, fbb_ts)
fbb_tau_tidx = get_time_idx(fbb_tau_times, fbb_ts)

@assert get_id.(fbb) == get_id.(fbb_tau)
@assert allequal([allequal(fbb_times[i] .== fbb_ts[i][fbb_ab_tidx[i]]) for i in 1:length(fbb)])
@assert allequal([allequal(fbb_tau_times[i] .== fbb_ts[i][fbb_tau_tidx[i]]) for i in 1:length(fbb_tau)])

fbb_suvr = calc_suvr.(fbb)
normalise!(fbb_suvr, fbb_u0, fbb_ui)
fbb_conc = map(x -> conc.(x, fbb_u0, fbb_ui), fbb_suvr)
fbb_inits = [d[:,1] for d in fbb_conc]

fbb_tau_suvr = calc_suvr.(fbb_tau)
normalise!(fbb_tau_suvr, v0, vi)
fbb_tau_conc = map(x -> conc.(x, v0, vi), fbb_tau_suvr)
fbb_tau_inits = [d[:,1] for d in fbb_tau_conc]

fbb_tau_pos_vol = get_vol.(fbb_tau)
fbb_total_vol_norm = [tp ./ sum(tp, dims=1) for tp in fbb_tau_pos_vol]
fbb_vols = [clamp.(1 .- (vol ./ vol[:,1]), 0, 1) for vol in fbb_total_vol_norm]
fbb_vol_inits = [vol[:,1] for vol in fbb_vols]

fbb_inits = [[ab; tau; vol] for (ab, tau, vol) in zip(fbb_inits, fbb_tau_inits, fbb_vol_inits)]
fbb_n = length(fbb)

fbb_atn_model = make_scaled_atn_model(fbb_ui .- fbb_u0, part .- v0, L)
fbb_prob = make_prob(fbb_atn_model, fbb_inits[1], (0.0,7.5), [1.0,0.1,1.0,3.5,1.0])
sol = solve(fbb_prob, Tsit5())

# --------------------------------------------------------------------------------
# Loading FBP data and aligning
# --------------------------------------------------------------------------------
tracer="FBP"
fbp_u0, fbp_ui = load_ab_params(tracer=tracer)
fbp_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
fbp_data = ADNIDataset(fbp_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

fbp, fbp_tau = align_data(fbp_data, tau_data; min_tau_scans=3)

fbp_times = get_times.(fbp)
fbp_tau_times = get_times.(fbp_tau)
fbp_ts = [sort(unique([a; t])) for (a, t) in zip(fbp_times, fbp_tau_times)]

fbp_ab_tidx = get_time_idx(fbp_times, fbp_ts)
fbp_tau_tidx = get_time_idx(fbp_tau_times, fbp_ts)

@assert get_id.(fbp) == get_id.(fbp_tau)
@assert allequal([allequal(fbp_times[i] .== fbp_ts[i][fbp_ab_tidx[i]]) for i in 1:length(fbp)])
@assert allequal([allequal(fbp_tau_times[i] .== fbp_ts[i][fbp_tau_tidx[i]]) for i in 1:length(fbp_tau)])

fbp_suvr = calc_suvr.(fbp)
normalise!(fbp_suvr, fbp_u0, fbp_ui)
fbp_conc = map(x -> conc.(x, fbp_u0, fbp_ui), fbp_suvr)
fbp_inits = [d[:,1] for d in fbp_conc]

fbp_tau_suvr = calc_suvr.(fbp_tau)
normalise!(fbp_tau_suvr, v0, vi)
fbp_tau_conc = map(x -> conc.(x, v0, vi), fbp_tau_suvr)
fbp_tau_inits = [d[:,1] for d in fbp_tau_conc]

fbp_tau_pos_vol = get_vol.(fbp_tau)
fbp_total_vol_norm = [tp ./ sum(tp, dims=1) for tp in fbp_tau_pos_vol]
fbp_vols = [clamp.(1 .- (vol ./ vol[:,1]), 0, 1) for vol in fbp_total_vol_norm]
fbp_vol_inits = [vol[:,1] for vol in fbp_vols]

fbp_inits = [[ab; tau; vol] for (ab, tau, vol) in zip(fbp_inits, fbp_tau_inits, fbp_vol_inits)]
fbp_n = length(fbp)

fbp_atn_model = make_scaled_atn_model(fbp_ui .- fbp_u0, part .- v0, L)
fbp_prob = make_prob(fbp_atn_model, fbp_inits[1], (0.0,7.5), [1.0,0.1,1.0,3.5,1.0])

pst = deserialize(projectdir("output/chains/population-scaled-atn/pst-samples-harmonised-dense-1x1000.jls"));
summarize(pst)
meanpst = mean(pst)

function get_diff(d)
    d[:,end] .- d[:,1]
end

# using GLMakie, Colors, ColorSchemes, Connectomes
# data_dashboard(tau, v0, vi; show_mtl_threshold=true)

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

        pstprob = ODEProblem(fbb_atn_model, fbb_inits[sub], (0.0, 10.0), p)
        pstsol = solve(pstprob, Tsit5())
        push!(ab_sols,pstsol(fbb_times[sub])[1:72,:])
        push!(tau_sols,pstsol(fbb_tau_times[sub])[73:144,:])
        push!(atr_sols,pstsol(fbb_tau_times[sub])[145:216,:])
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

    CairoMakie.scatter!(ax1, get_diff(fbb_suvr[sub]), get_diff(ab_sols[sub]))
    CairoMakie.scatter!(ax2, get_diff(fbb_tau_suvr[sub]), get_diff(tau_sols[sub]))
    CairoMakie.scatter!(ax3, get_diff(fbb_vols[sub]), get_diff(atr_sols[sub]))
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

    start = -0.0
    stop = 0.2
    border = 0.01
    ax4 =CairoMakie.Axis(f[1,1],  
            xlabel="Δ SUVR", 
            ylabel="Δ Prediction", 
            titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
            xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
            xticks=start:0.05:stop, yticks=start:0.05:stop, 
            xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
            xtickformat = "{:.3f}", ytickformat = "{:.3f}")

    CairoMakie.xlims!(ax4, start, stop + border)
    CairoMakie.ylims!(ax4, start, stop + border)
    lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
    start = -0.0
    stop = 0.2
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
    stop = 0.2
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
        meanpst[Symbol("α_t[$sub]"), :mean], meanpst[Symbol("β"), :mean], 
        meanpst[Symbol("η[$sub]"), :mean]]
        
        pstprob = remake(fbb_prob, u0=fbb_inits[sub], p=p)
        pstsol = solve(pstprob, Tsit5())
        push!(ab_sols,pstsol(fbb_times[sub])[1:72,:])
        push!(tau_sols,pstsol(fbb_tau_times[sub])[73:144,:])
        push!(atr_sols,pstsol(fbb_tau_times[sub])[145:216,:])
    end

    for (i, sub) in enumerate(23:44)
        p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
        meanpst[Symbol("α_t[$sub]"), :mean], meanpst[Symbol("β"), :mean], 
        meanpst[Symbol("η[$sub]"), :mean]]
        
        pstprob = remake(fbp_prob, u0=fbp_inits[i], p=p)
        pstsol = solve(pstprob, Tsit5())
        push!(ab_sols,  pstsol(fbp_times[i])[1:72,:])
        push!(tau_sols, pstsol(fbp_tau_times[i])[73:144,:])
        push!(atr_sols, pstsol(fbp_tau_times[i])[145:216,:])
    end

    mean_ab_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in ab_sols]), dims=2))
    mean_tau_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in tau_sols]), dims=2))
    mean_atr_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in atr_sols]), dims=2))
    
    mean_ab =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_conc; fbp_conc]]), dims=2))
    mean_tau =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_tau_conc; fbp_tau_conc]]), dims=2))
    mean_atr =  vec(mean(reduce(hcat, [d[:, end] for d in [fbb_vols; fbp_vols]]), dims=2))
    
    mean_ab_sol_diff = vec(mean(reduce(hcat, get_diff.(ab_sols)), dims=2))
    mean_tau_sol_diff = vec(mean(reduce(hcat, get_diff.(tau_sols)), dims=2))
    mean_atr_sol_diff = vec(mean(reduce(hcat, get_diff.(atr_sols)), dims=2))

    mean_ab_diff = vec(mean(reduce(hcat, get_diff.([fbb_conc; fbp_conc])), dims=2))
    mean_tau_diff = vec(mean(reduce(hcat, get_diff.([fbb_tau_conc; fbp_tau_conc])), dims=2))
    mean_atr_diff = vec(mean(reduce(hcat, get_diff.([fbb_vols; fbp_vols])), dims=2))

    labels = ["Braak 1", "Braak 2/3", "Braak 4", "Braak 5", "Braak 6"]
    for (i, rois) in enumerate(bs)
    CairoMakie.scatter!(ax4, mean_ab_diff[rois], mean_ab_sol_diff[rois], color=cmap[i], markersize=20 , label=labels[i])
    CairoMakie.scatter!(ax5, mean_tau_diff[rois], mean_tau_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
    CairoMakie.scatter!(ax6, mean_atr_diff[rois], mean_atr_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
    end
    Legend(f[2,:], ax6, framevisible = false, unique=true, labelsize=35, nbanks=5, tellheight=true)

    f
end
save(projectdir("output/plots/inference-results/pst-pred-harmonised-scaled.pdf"),f)

# pst = deserialize(projectdir("output/chains/population-scaled-atn/pst-samples-fixed-beta-$(tracer)-1x1000.jls"));
# using LsqFit
# linearmodel(x, p) = part .+ p[1] .* x
# fitted_model = curve_fit(linearmodel, ui .- u0, vi, [1.0])
# β = fitted_model.param[1]
# println("params = $(fitted_model.param)")

# function get_diff(d)
#     d[:,end] .- d[:,1]
# end

# using CairoMakie
# # begin    
# #     ab_sols = Vector{Array{Float64}}()
# #     tau_sols = Vector{Array{Float64}}()
# #     atr_sols = Vector{Array{Float64}}()
# #     # pst = deserialize(projectdir("analysis/output/chains/testatnpst-beta-u-conc-normal.jls"));
# #     meanpst = mean(pst)
# #     for sub in 1:22
# #     p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
# #     meanpst[Symbol("α_t[$sub]"), :mean], β, 
# #     meanpst[Symbol("η[$sub]"), :mean]]
# #     pstprob = ODEProblem(atn_model, inits[sub], (0.0, 10.0), p)
# #     pstsol = solve(pstprob, Tsit5())
# #     push!(ab_sols,pstsol(ab_times[sub])[1:72,:])
# #     push!(tau_sols,pstsol(tau_times[sub])[73:144,:])
# #     push!(atr_sols,pstsol(tau_times[sub])[145:216,:])
# #     end 
    
# #     for sub in 1:22
# #     f = Figure(size=(1500, 600))

# #     ax1 = CairoMakie.Axis(f[1,1])
# #     CairoMakie.xlims!(ax1, -0.5, 0.5)
# #     CairoMakie.ylims!(ax1, -0.5, 0.5)
    
# #     ax2 = CairoMakie.Axis(f[1,2])
# #     CairoMakie.xlims!(ax2, -1.0, 1.0)
# #     CairoMakie.ylims!(ax2, -1.0, 1.0)

# #     ax3 = CairoMakie.Axis(f[1,3])
# #     CairoMakie.xlims!(ax3, -0., 0.5)
# #     CairoMakie.ylims!(ax3, -0., 0.5)

# #     CairoMakie.scatter!(ax1, get_diff(ab_suvr[sub]), get_diff(ab_sols[sub]))
# #     CairoMakie.scatter!(ax2, get_diff(tau_suvr[sub]), get_diff(tau_sols[sub]))
# #     CairoMakie.scatter!(ax3, get_diff(vols[sub]), get_diff(atr_sols[sub]))
# #     display(f)
# #     end
# # end

# braak_regions = get_braak_regions()
# bs = [findall(x -> get_node_id(x) ∈ br, cortex) for br in braak_regions]

# begin
#     cmap = Makie.wong_colors();

#     f = Figure(size=(1500, 500))
#     titlesize = 40
#     xlabelsize = 25 
#     ylabelsize = 25
#     xticklabelsize = 20 
#     yticklabelsize = 20

#     start = -0.02
#     stop = 0.08
#     border = 0.005
#     ax4 =CairoMakie.Axis(f[1,1],  
#             xlabel="Δ SUVR", 
#             ylabel="Δ Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.02:stop, yticks=start:0.02:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.3f}", ytickformat = "{:.3f}")

#     CairoMakie.xlims!(ax4, start, stop + border)
#     CairoMakie.ylims!(ax4, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
#     start = -0.0
#     stop = 0.3
#     border = 0.01
#     ax5 =CairoMakie.Axis(f[1,2],  
#             xlabel="Δ SUVR", 
#             ylabel="Δ Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.05:stop, yticks=start:0.05:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     hideydecorations!(ax5, grid=false, ticks=false, ticklabels=false)
#     CairoMakie.xlims!(ax5, start, stop + border)
#     CairoMakie.ylims!(ax5, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)

#     start = -0.0
#     stop = 0.15
#     border = 0.01
#     ax6= CairoMakie.Axis(f[1,3],  
#             xlabel="Δ Atr.", 
#             ylabel="Δ Prediction", 
#             titlesize=titlesize, xlabelsize=xlabelsize, ylabelsize=ylabelsize, 
#             xticklabelsize=xticklabelsize, yticklabelsize=xticklabelsize,
#             xticks=start:0.05:stop, yticks=start:0.05:stop, 
#             xgridcolor=RGBAf(0, 0, 0, 0.15), ygridcolor=RGBAf(0, 0, 0, 0.15),
#             xtickformat = "{:.2f}", ytickformat = "{:.2f}")
#     hideydecorations!(ax6, grid=false, ticks=false, ticklabels=false)
#     CairoMakie.xlims!(ax6, start, stop + border)
#     CairoMakie.ylims!(ax6, start, stop + border)
#     lines!(start:0.01:stop+border, start:0.01:stop+border, color=(:grey, 1.0), linewidth=5, linestyle=:dash)
    
#     ab_sols = Vector{Array{Float64}}()
#     tau_sols = Vector{Array{Float64}}()
#     atr_sols = Vector{Array{Float64}}()
#     meanpst = mean(pst)

#     for sub in 1:22
#         p = [meanpst[Symbol("α_a[$sub]"), :mean], meanpst[Symbol("ρ_t[$sub]"), :mean], 
#         meanpst[Symbol("α_t[$sub]"), :mean], β, 
#         meanpst[Symbol("η[$sub]"), :mean]]
        
#         pstprob = ODEProblem(atn_model, inits[sub], (0.0, 10.0), p)
#         pstsol = solve(pstprob, Tsit5())
#         push!(ab_sols,pstsol(ab_times[sub])[1:72,:])
#         push!(tau_sols,pstsol(tau_times[sub])[73:144,:])
#         push!(atr_sols,pstsol(tau_times[sub])[145:216,:])
#     end
#     mean_ab_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in ab_sols]), dims=2))
#     mean_tau_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in tau_sols]), dims=2))
#     mean_atr_sols =  vec(mean(reduce(hcat, [sol[:, end] for sol in atr_sols]), dims=2))
    
#     mean_ab =  vec(mean(reduce(hcat, [d[:, end] for d in ab_suvr]), dims=2))
#     mean_tau =  vec(mean(reduce(hcat, [d[:, end] for d in tau_suvr]), dims=2))
#     mean_atr =  vec(mean(reduce(hcat, [d[:, end] for d in vols]), dims=2))
    
#     mean_ab_sol_diff = vec(mean(reduce(hcat, get_diff.(ab_sols)), dims=2))
#     mean_tau_sol_diff = vec(mean(reduce(hcat, get_diff.(tau_sols)), dims=2))
#     mean_atr_sol_diff = vec(mean(reduce(hcat, get_diff.(atr_sols)), dims=2))

#     mean_ab_diff = vec(mean(reduce(hcat, get_diff.(ab_suvr)), dims=2))
#     mean_tau_diff = vec(mean(reduce(hcat, get_diff.(tau_suvr)), dims=2))
#     mean_atr_diff = vec(mean(reduce(hcat, get_diff.(vols)), dims=2))

#     labels = ["Braak 1", "Braak 2/3", "Braak 4", "Braak 5", "Braak 6"]
#     for (i, rois) in enumerate(bs)
#     CairoMakie.scatter!(ax4, mean_ab_diff[rois], mean_ab_sol_diff[rois], color=cmap[i], markersize=20 , label=labels[i])
#     CairoMakie.scatter!(ax5, mean_tau_diff[rois], mean_tau_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
#     CairoMakie.scatter!(ax6, mean_atr_diff[rois], mean_atr_sol_diff[rois], color=cmap[i], markersize=20, label=labels[i])
#     end
#     Legend(f[2,:], ax6, framevisible = false, unique=true, labelsize=35, nbanks=5, tellheight=true)

#     f
# end
# save(projectdir("output/plots/inference-results/pst-pred-$(tracer)-fixed-beta-scaled.pdf"),f)
