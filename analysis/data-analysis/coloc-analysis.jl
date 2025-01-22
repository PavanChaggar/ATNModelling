using ATNModelling.SimulationUtils: make_prob, make_scaled_atn_model, 
                                    simulate, resimulate, simulate_amyloid,
                                    load_ab_params, load_tau_params, conc
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex,
                                    get_dkt_names, get_braak_regions
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise
using ATNModelling.InferenceModels: fit_model, ensemble_atn_truncated, serial_atn, fit_serial_atn

using Connectomes: laplacian_matrix, get_label, get_node_id
using ADNIDatasets: ADNIDataset, get_id, get_dates, get_initial_conditions, calc_suvr, get_vol, get_times
using DrWatson: projectdir, datadir
using CSV, DataFrames


# --------------------------------------------------------------------------------
# Loading data and aligning
# --------------------------------------------------------------------------------
v0, vi, part = load_tau_params()
c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c) 
cortex = get_parcellation() |> get_cortex 
dktnames = get_dkt_names(cortex)

braak_regions = get_braak_regions()
bs = [findall(x -> get_node_id(x) ∈ br, cortex) for br in braak_regions]
bs_idx = reduce(vcat, bs)

begin
    f = Figure(size=(800, 500))
    mag = Vector{Float64}()
    for (i, tracer) in enumerate(["FBB", "FBP"])
        u0, ui = load_ab_params(tracer=tracer)
        # Amyloid data 
        _ab_data_df =  CSV.read(datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv"), DataFrame)
        _tau_data_df = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame) 

        ab_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
        # ab_data_df = filter(x -> x.qc_flag==2, _ab_data_df);
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

        @assert get_id.(ab) == get_id.(tau)
        @assert allequal([allequal(ab_times[i] .== ts[i][ab_tidx[i]]) for i in 1:length(ab)])
        @assert allequal([allequal(tau_times[i] .== ts[i][tau_tidx[i]]) for i in 1:length(tau)])

        ab_suvr = calc_suvr.(ab)
        normalise!(ab_suvr, u0, ui)
        ab_conc = map(x -> conc.(x, u0, ui), ab_suvr)
        ab_inits = [d[:,1] for d in ab_conc]

        tau_suvr = calc_suvr.(tau)
        normalise!(tau_suvr, v0, vi)
        tau_conc = map(x -> conc.(x, v0, vi), tau_suvr)
        tau_inits = [d[:,1] for d in tau_conc] 

        ax = Axis(f[1,i], title=tracer, titlefont=:regular, titlesize=25)
        # coloc = [sqrt.(a.^2 .+ t.^2) for (a, t) in zip(ab_inits, tau_inits)]
        coloc = [a .* t for (a, t) in zip(ab_inits, tau_inits)]
        idxs = reverse(sortperm(sum.(coloc)))
        coloc_array = reduce(hcat, coloc[idxs])
        heatmap!(coloc_array[bs_idx,:] ./ maximum(coloc_array))
        ax = Axis(f[2,i])
        hideydecorations!(ax)
        avg_mag = [mean(coloc_array[b,:1:22]) for b in bs]    
        println(avg_mag)
        heatmap!(reshape(avg_mag, 1, 5)', colorrange=(0,0.31))   
        centers_x = 1:5
        centers_y = ones(5)
        text!(centers_x, centers_y, text = string.(round.(avg_mag, digits=2)), align=(:center, :center))
        rowsize!(f.layout, 2, 50)
    end
    f
end
save(projectdir("output/plots/population-analysis/coloclisation.pdf"), f)