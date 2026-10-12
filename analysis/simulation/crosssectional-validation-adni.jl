# Combined ADNI FBB/FTP and FBP/FTP cross-sectional Braak-stage validation.
# Minimal refactor of longitudinal-validation-fbb.jl / longitudinal-validation-fbp.jl.
#
# For each tracer and Braak stage (1:5), saves:
#   - subject_ab_sols, subject_tau_sols: T × N posterior-predictive subject
#     trajectories (grey lines)
#   - meanparam_ab_sols, meanparam_tau_sols: T × N trajectories obtained
#     using expected subject-level parameters at each hyperposterior draw (blue)
#   - reference_ab_sol, reference_tau_sol: T-vectors simulated using the
#     posterior means of expected subject-level parameters (orange)
#   - observations: actual paired FBB/FBP and FTP measurements, within 1 yr
#     (green points). Full regional data are also saved once per tracer.
#
# All simulation matrices have time on rows and posterior simulations on columns;
# matched amyloid/tau columns share the same simulated subject / posterior index.
# The model ODE is solved once per parameter draw and tracer, then summarized
# separately by Braak stage to avoid 5× redundant simulations.
#
# Run: julia --project=. plots/data-analysis/longitudinal-validation-both-tracers-braak.jl

using ATNModelling.SimulationUtils: make_atn_model, simulate, load_ab_params, load_tau_params
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names, get_braak_regions
using ATNModelling.DataUtils: align_data, normalise!
using ADNIDatasets: ADNIDataset, get_id, calc_suvr, get_dates
using Connectomes: laplacian_matrix, get_node_id, get_label
using DrWatson: projectdir, datadir
using CSV, DataFrames, Dates, Statistics, Serialization, Random, Distributions, Turing, SciMLBase
using DelimitedFiles
using CairoMakie

NREG = 72
TRACERS = ("FBB", "FBP")
INIT_CL = (25.0, 55.0)
VALIDATION_CL = (40.0, 300.0)
INIT_DT = 0.3              # Original scripts; align_data dt in years
OBS_DT = 0.6               # Paired PET scans within one year
TIMES = collect(0.0:0.25:60.0)
DRAW_INDICES = collect(1:4:4000) # 1000 paired posterior samples, as before
OUTPUT_ROOT = projectdir("output/validation/longitudinal-braak")
AMY_FILE = datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv")
TAU_FILE = datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv")
CHAIN_PATTERN = "output/chains/population-atn/pst-samples-harmonised-suvr-random-beta-lognormal-4x1000-%d.jls"

function chain_vector(ch, name::Symbol)
    vec(Array(ch[:, [name], :]))
end

# Identical expected-subject-parameter calculation to the original scripts.
function posterior_subject_mean(ch, mean_param, sd_param)
    mu, sd = chain_vector(ch, mean_param), chain_vector(ch, sd_param)
    [mean(truncated(Normal(m, s), 0.0, Inf)) for (m, s) in zip(mu, sd)]
end

# Convert the matched ADNIDataset subjects into a regional matrix in their
# common ordering. The original scripts use the FIRST aligned visit per subject.
function aligned_observations(amy_subs, tau_subs)
    @assert length(amy_subs) == length(tau_subs)
    @assert get_id.(amy_subs) == get_id.(tau_subs) "Amyloid/tau order differs"
    ids = Int.(get_id.(amy_subs))
    ab = reduce(hcat, [vec(calc_suvr(sub)[:, 1]) for sub in amy_subs])
    tau = reduce(hcat, [vec(calc_suvr(sub)[:, 1]) for sub in tau_subs])
    @assert size(ab, 1) == NREG && size(tau, 1) == NREG
    # Dates stored as strings so this works with Date or DateTime return types.
    amy_dates = [string(first(get_dates(sub))) for sub in amy_subs]
    tau_dates = [string(first(get_dates(sub))) for sub in tau_subs]
    return ids, amy_dates, tau_dates, ab, tau
end

function main()
    mkpath(OUTPUT_ROOT)
    amy_raw = CSV.read(AMY_FILE, DataFrame)
    tau_raw = CSV.read(TAU_FILE, DataFrame)
    tau_ids = Set(skipmissing(tau_raw.RID))
    v0, vi, part = load_tau_params()
    cortex = get_parcellation() |> get_cortex
    dktnames = get_dkt_names(cortex)
    L = laplacian_matrix(get_connectome(; include_subcortex=false,
                                       apply_filter=true, filter_cutoff=1e-2))

    # IMPORTANT: Braak atlas IDs are not indices into the 72-region matrix.
    # Here bs[1] corresponds to stage 1, ... bs[5] to stage 5, as requested.
    bs = get_braak_regions()
    @assert length(bs) >= 5 "Expected at least five Braak groups"
    stage_indices = Dict(stage => findall(r -> get_node_id(r) in Set(bs[stage]), cortex)
                         for stage in 1:5)
    for stage in 1:5
        @assert !isempty(stage_indices[stage]) "No regions for Braak stage $stage"
        @info "Braak stage regions" stage n=length(stage_indices[stage]) labels=get_label.(cortex[stage_indices[stage]])
    end

    paths = [projectdir(replace(CHAIN_PATTERN, "%d" => string(i))) for i in 1:4]
    @assert all(isfile, paths) "Missing posterior chain files"
    pst = chainscat(deserialize.(paths)...)
    @assert maximum(DRAW_INDICES) <= length(chain_vector(pst, :Am_a))

    # Shared posterior hyperparameters across tracers; beta is tracer-specific.
    hypers = [
        (chain_vector(pst, :Am_a), chain_vector(pst, :As_a)),
        (chain_vector(pst, :Pm_t), chain_vector(pst, :Ps_t)),
        (chain_vector(pst, :Am_t), chain_vector(pst, :As_t)),
        (chain_vector(pst, :Em), chain_vector(pst, :Es)),
    ]
    expected = [posterior_subject_mean(pst, p, s) for (p, s) in
                ((:Am_a, :As_a), (:Pm_t, :Ps_t), (:Am_t, :As_t), (:Em, :Es))]

    # Build a single FTP tau initialisation profile from the union of participants
    # with FBB or FBP amyloid scans in INIT_CL and an FTP scan within INIT_DT.
    # Retain the *aligned* FTP visit, not an arbitrary first FTP visit.
    # A RID qualifying under both tracers contributes just once. In that case
    # the first matched visit under TRACERS order is retained.
    shared_tau_profiles = Dict{Int, Vector{Float64}}()
    shared_tau_source = Dict{Int, String}()
    for initial_tracer in TRACERS
        early_ab_df = filter(x -> x.qc_flag == 2 &&
                                  x.TRACER == initial_tracer &&
                                  !ismissing(x.CENTILOIDS) &&
                                  x.RID in tau_ids &&
                                  INIT_CL[1] < x.CENTILOIDS < INIT_CL[2], amy_raw)
        early_ab = ADNIDataset(early_ab_df, dktnames;
                               min_scans=1, reference_region="COMPOSITE_REF")
        early_tau_df = filter(x -> x.qc_flag == 2 &&
                                   x.TRACER == "FTP" &&
                                   x.RID in Set(get_id.(early_ab)), tau_raw)
        early_tau = ADNIDataset(early_tau_df, dktnames; min_scans=1)
        early_ab, early_tau = align_data(early_ab, early_tau;
                                         min_tau_scans=1, dt=INIT_DT)
        @assert get_id.(early_ab) == get_id.(early_tau)
        for sub in early_tau
            rid = Int(get_id(sub))
            if !haskey(shared_tau_profiles, rid)
                profile = Float64.(vec(calc_suvr(sub)[:, 1]))
                @assert length(profile) == NREG && all(isfinite, profile)
                shared_tau_profiles[rid] = profile
                shared_tau_source[rid] = initial_tracer
            end
        end
        @info "Shared tau cohort: paired scans" tracer=initial_tracer n=length(early_tau)
    end
    isempty(shared_tau_profiles) && error("No paired FTP scans for shared tau initialisation")
    shared_rids = sort!(collect(keys(shared_tau_profiles)))
    shared_tau_suvr = [reshape(copy(shared_tau_profiles[rid]), NREG, 1)
                       for rid in shared_rids]
    normalise!(shared_tau_suvr, v0, vi)  # SUVR clamping, not rescaling
    shared_tau_inits = vec(mean(reduce(hcat, [vec(d) for d in shared_tau_suvr]), dims=2))
    tau_cutoffs = vec(readdlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-1std.csv")))

    shared_tau_inits[shared_tau_inits .< tau_cutoffs] .= v0[shared_tau_inits .< tau_cutoffs]
    @assert length(shared_tau_inits) == NREG && all(isfinite, shared_tau_inits)
    CSV.write(joinpath(OUTPUT_ROOT, "shared_tau_initialisation_subjects.csv"),
              DataFrame(RID=shared_rids,
                        selected_from_tracer=[shared_tau_source[rid] for rid in shared_rids]))
    serialize(joinpath(OUTPUT_ROOT, "shared_tau_initial_conditions.jls"),
              (RID=shared_rids, tau_inits=shared_tau_inits,
               tau_matrix=reduce(hcat, [vec(d) for d in shared_tau_suvr])))
    @info "Shared FTP initial profile created" n=length(shared_rids) mean_tau=mean(shared_tau_inits)

    for tracer in TRACERS
        @info "Processing tracer" tracer
        dir = joinpath(OUTPUT_ROOT, lowercase(tracer))
        mkpath(dir)
        u0, ui = load_ab_params(tracer=tracer)
        beta = chain_vector(pst, Symbol("β_" * lowercase(tracer)))
        rng = MersenneTwister(42)  # Reproducible per tracer

        # --- Early-stage initialisation (same logic as input scripts) ---
        init_ab_df = filter(x -> x.qc_flag == 2 && x.TRACER == tracer &&
                                 !ismissing(x.CENTILOIDS) &&
                                 x.RID in tau_ids &&
                                 INIT_CL[1] < x.CENTILOIDS < INIT_CL[2], amy_raw)
        init_ab = ADNIDataset(init_ab_df, dktnames;
                              min_scans=1, reference_region="COMPOSITE_REF")
        init_tau_df = filter(x -> x.qc_flag == 2 && x.TRACER == "FTP" &&
                                  x.RID in Set(get_id.(init_ab)), tau_raw)
        init_tau = ADNIDataset(init_tau_df, dktnames; min_scans=1)
        init_ab, init_tau = align_data(init_ab, init_tau;
                                       min_tau_scans=1, dt=INIT_DT)
        isempty(init_ab) && error("No $tracer initialisation subjects")
        ab_suvr = calc_suvr.(init_ab)
        normalise!(ab_suvr, u0, ui)
        ab_inits = mean([d[:, 1] for d in ab_suvr])
        # Identical regional FTP tau initial conditions for FBB and FBP.
        # Tracer-specific paired visits are still used to estimate amyloid initials.
        tau_inits = copy(shared_tau_inits)
        init_state = [ab_inits; tau_inits; zeros(NREG)]
        @assert length(init_state) == 3NREG

        # After initialisation for each tracer:
        println("Tracer: ", tracer)

        println("Mean initial amyloid: ", mean(ab_inits))
        println("Mean initial tau: ", mean(tau_inits))

        println("Mean amyloid baseline: ", mean(u0))
        println("Mean amyloid capacity: ", mean(ui))
        println("Mean amyloid dynamic range: ", mean(ui .- u0))

        println("Mean initial amyloid fraction: ",
            mean((ab_inits .- u0) ./ (ui .- u0)))

        println("Mean tau above baseline, Braak I: ",
            mean(tau_inits[stage_indices[1]] .-
                v0[stage_indices[1]]))

        println("Mean tau above baseline, Braak II: ",
            mean(tau_inits[stage_indices[2]] .-
                v0[stage_indices[2]]))
        display(scatter(tau_inits .- v0))
        println("Posterior mean beta: ", mean(beta))
        # --- Observed PET pairs within one year (no disease-time inference) ---
        obs_ab_df = filter(x -> x.qc_flag == 2 && x.TRACER == tracer &&
                                 !ismissing(x.CENTILOIDS) &&
                                 x.RID in tau_ids &&
                                 VALIDATION_CL[1] < x.CENTILOIDS < VALIDATION_CL[2], amy_raw)
        obs_ab = ADNIDataset(obs_ab_df, dktnames;
                             min_scans=1, reference_region="COMPOSITE_REF")
        obs_tau_df = filter(x -> x.qc_flag == 2 && x.TRACER == "FTP" &&
                                  x.RID in Set(get_id.(obs_ab)), tau_raw)
        obs_tau = ADNIDataset(obs_tau_df, dktnames; min_scans=1)
        obs_ab, obs_tau = align_data(obs_ab, obs_tau;
                                      min_tau_scans=1, dt=OBS_DT)
        isempty(obs_ab) && error("No matched observations for $tracer")
        ids, ab_dates, tau_dates, obs_ab_matrix, obs_tau_matrix =
            aligned_observations(obs_ab, obs_tau)
        # Retain the original scripts' exact data transformation pathway:
        # normalise! acts on the vector of regional scan matrices.
        obs_ab_suvr = calc_suvr.(obs_ab)
        normalise!(obs_ab_suvr, u0, ui)
        obs_tau_suvr = calc_suvr.(obs_tau)
        normalise!(obs_tau_suvr, v0, vi)
        obs_ab_matrix = reduce(hcat, [d[:, 1] for d in obs_ab_suvr])
        obs_tau_matrix = reduce(hcat, [d[:, 1] for d in obs_tau_suvr])
        # Save complete regional measurements to support later plotting choices.
        serialize(joinpath(dir, "observations_regional.jls"),
                  (RID=ids, amyloid_dates=ab_dates, tau_dates=tau_dates,
                   amyloid=obs_ab_matrix, tau=obs_tau_matrix))
        @info "Matched observed participants" tracer n=length(ids)

        # --- Simulations: save entire 72-region state once per draw ---
        model = make_atn_model(u0, ui, v0, part, L)
        ntime = length(TIMES)
        ndraw = length(DRAW_INDICES)
        ab_subject = Array{Float64}(undef, NREG, ntime, ndraw)
        tau_subject = Array{Float64}(undef, NREG, ntime, ndraw)
        ab_meanparam = similar(ab_subject)
        tau_meanparam = similar(tau_subject)

        ref_params = [mean(expected[1]), mean(expected[2]), mean(expected[3]),
                      mean(beta), mean(expected[4])]
        ref_sol = simulate(model, init_state, (0.0, 60.0), ref_params;
                           saveat=TIMES, tol=1e-6)
        @assert SciMLBase.successful_retcode(ref_sol.retcode)
        ref_array = Array(ref_sol)

        for (k, s) in enumerate(DRAW_INDICES)
            mean_params = [expected[1][s], expected[2][s], expected[3][s],
                           beta[s], expected[4][s]]
            subject_params = [
                rand(rng, truncated(Normal(hypers[1][1][s], hypers[1][2][s]), 0.0, Inf)),
                rand(rng, truncated(Normal(hypers[2][1][s], hypers[2][2][s]), 0.0, Inf)),
                rand(rng, truncated(Normal(hypers[3][1][s], hypers[3][2][s]), 0.0, Inf)),
                beta[s],
                rand(rng, truncated(Normal(hypers[4][1][s], hypers[4][2][s]), 0.0, Inf)),
            ]
            sol_subject = simulate(model, init_state, (0.0, 60.0), subject_params;
                                   saveat=TIMES, tol=1e-6)
            sol_mean = simulate(model, init_state, (0.0, 60.0), mean_params;
                                saveat=TIMES, tol=1e-6)
            @assert SciMLBase.successful_retcode(sol_subject.retcode) "Subject simulation failed: $tracer / $s"
            @assert SciMLBase.successful_retcode(sol_mean.retcode) "Mean-param simulation failed: $tracer / $s"
            arr_subject = Array(sol_subject)
            arr_mean = Array(sol_mean)
            ab_subject[:, :, k] = arr_subject[1:NREG, :]
            tau_subject[:, :, k] = arr_subject[NREG+1:2NREG, :]
            ab_meanparam[:, :, k] = arr_mean[1:NREG, :]
            tau_meanparam[:, :, k] = arr_mean[NREG+1:2NREG, :]
        end

        # Global amyloid vs stage-specific tau: each matched column of the
        # matrices describes the SAME simulated hypothetical subject/draw.
        global_subject = dropdims(mean(ab_subject, dims=1), dims=1)
        global_meanparam = dropdims(mean(ab_meanparam, dims=1), dims=1)
        global_ref = vec(mean(ref_array[1:NREG, :], dims=1))

        for stage in 1:5
            roi = stage_indices[stage]
            # Both sets of matrices are time × posterior sample.
            subject_tau = dropdims(mean(tau_subject[roi, :, :], dims=1), dims=1)
            meanparam_tau = dropdims(mean(tau_meanparam[roi, :, :], dims=1), dims=1)
            ref_tau = vec(mean(ref_array[NREG .+ roi, :], dims=1))
            observed_tau = vec(mean(obs_tau_matrix[roi, :], dims=1))
            observed_ab = vec(mean(obs_ab_matrix, dims=1))

            observations = DataFrame(RID=ids, amyloid_date=ab_dates, tau_date=tau_dates,
                                     amyloid_suvr=observed_ab, tau_suvr=observed_tau)
            CSV.write(joinpath(dir, "braak$(stage)_observations.csv"), observations)
            serialize(joinpath(dir, "braak$(stage)_trajectories.jls"), (
                tracer=tracer,
                stage=stage,
                region_indices=roi,
                region_labels=get_label.(cortex[roi]),
                times=TIMES,
                draw_indices=DRAW_INDICES,
                subject_ab_sols=global_subject,
                subject_tau_sols=subject_tau,
                meanparam_ab_sols=global_meanparam,
                meanparam_tau_sols=meanparam_tau,
                reference_ab_sol=global_ref,
                reference_tau_sol=ref_tau,
                observations=observations,
                initial_CL_range=INIT_CL,
                observations_CL_range=VALIDATION_CL,
                paired_within_years=OBS_DT,
            ))
            @info "Saved Braak stage" tracer stage n_subjects=nrow(observations) n_draws=ndraw
        end
    end
    @info "Finished" OUTPUT_ROOT
end

main()