module DataUtils

using ADNIDatasets: ADNIDataset, ADNISubject, calc_suvr, get_times, 
                    get_id, get_initial_conditions, get_dates
using Statistics: mean
using Polynomials: fit, roots
using NonlinearSolve: NonlinearProblem, solve, NewtonRaphson
using DataFrames: DataFrame
using LsqFit: curve_fit, LsqFitResult
using Distributions: Normal

"""
    baseline_difference(data::ADNIDataset, region::Int)

Calculate the longitudinal difference between first and last scan SUVR for the given region.
"""
function baseline_difference(data::ADNIDataset, region::Int)
    vals = Vector{Float64}()
    diffs = Vector{Float64}()
    
    for i in 1:length(data)
        sd = calc_suvr(data, i)

        _vals = sd[region,:] 
        _times = get_times(data, i)
        
        val_diff = _vals[end] - _vals[1]
        t_diff = _times[end] - _times[1]

        diff = val_diff / t_diff
        
        push!(vals, _vals[1])
        push!(diffs, diff)
    end
    return DataFrame(ab_suvr = vals, ab_diff = diffs)
end

function interval(vals, start, stop)
    findall(x -> start <= x < stop, vals)
end


"""
    split_data(suvr::Vector{Float64}, delta_suvr::Vector{Float64}, start, step, stop)

Bin SUVR data along interval start:step:stop and calculate the mean SUVR change for the interval.
"""
function split_data(suvr::Vector{Float64}, delta_suvr::Vector{Float64}, start, step, stop)

    @assert length(suvr) == length(delta_suvr)
    
    ints = Vector{Vector{Float64}}()
    ps = collect(start:step:stop)
    
    starts = findall(x -> x < start, suvr)
    push!(ints, delta_suvr[starts])
    
    for i in ps
        idxs = interval(suvr, i, i+step)
        push!(ints, delta_suvr[idxs])
    end

    stops = findall(x -> x > stop, suvr)
    push!(ints, delta_suvr[stops])

    xs = [mean(suvr[starts]) ; collect(start+step/2:step:stop+step); mean(suvr[stops])] 
    return DataFrame(ab_bin = xs, ab_bin_diffs = mean.(ints))
end

"""
    fit_second_order_polynomial(vals::Vector{Float64}, delta_suvr::Vector{Float64})

Fit a second order polynomial to phase space data given by (suvr, delta_suvr).
"""
function fit_second_order_polynomial(suvr::Vector{Float64}, delta_suvr::Vector{Float64})
    f = fit(suvr, delta_suvr, 2)
    return f, roots(f)
end
function fit_second_order_polynomial(suvr::Vector{Float64}, delta_suvr::Vector{Vector{Float64}})
    return fit_second_order_polynomial(suvr, mean.(delta_suvr))
end

"""
    solve_amyloid_time(xt::Function, suvr::Vector{Float64})

Find root of function polynomial function xt given SUVR value
"""
function solve_amyloid_time(xt::Function, suvr::Vector{Float64})
    ts = Vector{Float64}()
    for val in suvr
        probN = NonlinearProblem(xt, 40.0, val)
        sol = solve(probN, NewtonRaphson(), reltol = 1e-9)
        push!(ts, sol.u)
    end
    return ts
end

"""
    find_amyloid_time(xt::Function, data::ADNIDataset)

Find root of function xt given SUVR value for each subject in `data`. Assumes 
the final SUVR value corresponds to the amyloid cortical summary.
"""
function find_amyloid_time(xt::Function, data::ADNIDataset)
    vals = [d[end] for d in get_initial_conditions.(data)]
    ts = solve_amyloid_time(xt, vals)
    ts_idx = findall( x -> x > 0 && x < 100, ts)

    return DataFrame(sub_id = get_id.(data[ts_idx]), 
                     t_idx = ts_idx, 
                     ab_time = ts[ts_idx],
                     ab_summary=vals[ts_idx])

end


"""
    sigmoid(t, p)

Returns a sigmoid function at time `t` with parameters `p`, where 
p is a 4-dimensional vector comprising the carrying capacity, production 
rate, t50 and baseline value.
"""
sigmoid(t, p) = @. p[1] / (1 + exp(-p[2]*(t - p[3]))) + p[4]

"""
    find_regional_params(data, t_df)

Find regional parameters for sigmoid function given the data and temporal indices according to amyloid time.
"""
function find_regional_params(data::ADNIDataset, t_df::DataFrame)
    ts = t_df.ab_time
    t_idx = t_df.t_idx
    params = Vector{LsqFitResult}()
    for i in 1:72
        ab_df = baseline_difference(data[t_idx], i)
        roi_vals = ab_df.ab_suvr

        _p0 = mean(sort(roi_vals)[1:100])
        _pi = mean(sort(roi_vals)[600:end]) .- _p0
        p0 = [_pi,1.0,40.0,_p0]
        fitted_model = curve_fit(sigmoid, ts, roi_vals, p0)
        push!(params, fitted_model)
    end
    return params
end

"""
    set_ab_status(ab::DataFrame, tau::DataFrame)

Assign an amyloid status to each sacn in the tau DataFrame. 
"""
function set_ab_status(ab::DataFrame, tau::DataFrame)
    tau_df = deepcopy(tau)
    abstatus = ab[:, "AMYLOID_STATUS_COMPOSITE_REF"] 
    ab[:, "AMYLOID_STATUS_COMPOSITE_REF"] = coalesce.(ab[:,"AMYLOID_STATUS_COMPOSITE_REF"], -1) 

    zipped_abstatus = zip(findall(x -> x isa Int && x == 1, abstatus), 
                      findall(x -> x isa Int && x == 1, ab[:, "AMYLOID_STATUS_COMPOSITE_REF"])) 

    @assert allequal(allequal.(zipped_abstatus))

    tau_df.AB_Status = fill(-1, size(tau_df, 1)) 

    for scan in eachrow(tau_df) # Iterate through each row of tau data
        ID = scan.RID # Get the RID (subject ID) for the current tau row
        ab_df = filter(x -> x.RID == ID, ab) # Filter amyloid data for matching RID
        if size(ab_df, 1) == 0 # If no matching amyloid data exists
            scan.AB_Status = -1 # Set status to -1
        else
            tau_scan_date = scan.SCANDATE # Get the scan date for the tau row
            ab_scan_dates = ab_df.SCANDATE # Get all scan dates for the matching amyloid rows
            nearest_ab_scan_idx = argmin(abs.(ab_scan_dates .- tau_scan_date)) # Find index of the nearest scan date
            scan.AB_Status = ab_df[nearest_ab_scan_idx, "AMYLOID_STATUS"] # Assign corresponding amyloid status
        end
    end
    return tau_df
end

"""
    set_tau_status(tau::DataFrame, df_names, mtl_idx, neo_idx, mtl_cutoff, neo_cutoff)

Assign MTL and Neocortical status for each tau PET scan in tau. 
"""
function set_tau_status(tau::DataFrame, df_names, mtl_idx, neo_idx, mtl_cutoff, neo_cutoff)
    taudata = deepcopy(tau)
    taudata.MTL_Status = fill(-1, size(taudata, 1)) # Initialize "MTL_Status" column with -1
    taudata.NEO_Status = fill(-1, size(taudata, 1)) # Initialize "NEO_Status" column with -1

    for scan in eachrow(taudata) # Iterate through each row of tau data
        mtl = mean(Array(scan[df_names[mtl_idx]])) # Compute mean SUVR for medial temporal lobe regions
        neo = mean(Array(scan[df_names[neo_idx]])) # Compute mean SUVR for neocortical regions
        if mtl isa Missing # Check for missing MTL data
            scan.MTL_Status = -1 # Set status to -1 if data is missing
        elseif mtl >= mtl_cutoff # Check if MTL mean exceeds cutoff
            scan.MTL_Status = 1 # Set status to 1 (positive)
        else
            scan.MTL_Status = 0 # Set status to 0 (negative)
        end

        if neo isa Missing # Check for missing neocortical data
            scan.NEO_Status = -1 # Set status to -1 if data is missing
        elseif neo >= neo_cutoff # Check if neocortical mean exceeds cutoff
            scan.NEO_Status = 1 # Set status to 1 (positive)
        else
            scan.NEO_Status = 0 # Set status to 0 (negative)
        end
    end
    return taudata
end

"""
    get_dkt_moments(gmm_moments::DataFrame, dktnames)

Get moments from Gaussian mixture model output. 
"""
function get_dkt_moments(gmm_moments::DataFrame, dktnames)
    μ_1 = Vector{Float64}()
    μ_2 = Vector{Float64}()

    σ_1 = Vector{Float64}()
    σ_2 = Vector{Float64}()
    
    for name in dktnames[1:end]
        roi = filter( x -> x.region == name, gmm_moments)
        if roi.C0_mean[1] < roi.C1_mean[1]
            push!(μ_1, roi.C0_mean[1])
            push!(μ_2, roi.C1_mean[1])
            push!(σ_1, sqrt(roi.C0_cov[1]))
            push!(σ_2, sqrt(roi.C1_cov[1]))
        elseif roi.C0_mean[1] > roi.C1_mean[1]
            push!(μ_1, roi.C1_mean[1])
            push!(μ_2, roi.C0_mean[1])
            push!(σ_1, sqrt(roi.C1_cov[1]))
            push!(σ_2, sqrt(roi.C0_cov[1]))
        end
    end
    Normal.(μ_1, σ_1), Normal.(μ_2, σ_2)
end

"""
    normalise!(data::Vector{Matrix{Float64}}, lower::Vector{Float64}, upper::Vector{Float64})

Normalise in-place the regional data to lie between a `lower` and `upper` 
bound.
"""
function normalise!(data::Vector{Matrix{Float64}}, lower::Vector{Float64}, upper::Vector{Float64})
    @assert length(lower) == length(upper)
    @assert allequal(size.(data, 1) .== length(lower))
    for d in data
        for i in axes(d, 1)
            lower_mask = d[i,:] .< lower[i]
            d[i, lower_mask] .= lower[i]
            upper_mask = d[i,:] .> upper[i]
            d[i, upper_mask] .= upper[i]
        end
    end
    @assert allequal([allequal(d .>= lower) for d in data])
    @assert allequal([allequal(d .<= upper) for d in data])
end

function normalise!(data::Vector{Matrix{Float64}}, lower::Vector{Float64})
    @assert allequal(size.(data, 1) .== length(lower))
    for d in data
        for i in axes(d, 1)
            lower_mask = d[i,:] .< lower[i]
            d[i, lower_mask] .= lower[i]
        end
    end
    @assert allequal([allequal(d .>= lower) for d in data])
end

function calc_times(ab::ADNISubject, tau::ADNISubject)
    ab_dates = get_dates(ab)
    tau_dates = get_dates(tau)
    min_date = minimum([ab_dates; tau_dates])
    _ab_dates = ab_dates .- min_date
    _tau_dates = tau_dates .- min_date
    ab_times = [d.value for d in _ab_dates] ./ 365
    tau_times = [d.value for d in _tau_dates] ./ 365
    return ab_times, tau_times
end

function calc_times(ab::ADNIDataset, tau::ADNIDataset)
    ab_times = Vector{Vector{Float64}}()
    tau_times = Vector{Vector{Float64}}()

    for (a, t) in zip(ab, tau)
        _ab_times, _tau_times = calc_times(a, t)
        push!(ab_times, _ab_times)
        push!(tau_times, _tau_times)
    end
    return ab_times, tau_times
end

function align_data(ab_data, tau_data; min_tau_scans=3)
    pos_ids = get_id.(tau_data)
    ab_tau_pos = filter(x -> get_id(x) ∈ pos_ids, ab_data)
    ab_tau_pos_ids = get_id.(ab_tau_pos)
    
    tau_pos = filter(x -> get_id(x) ∈ ab_tau_pos_ids, tau_data)
    sub_idx = reduce(vcat, [findall(x -> get_id(x) ∈ id, tau_pos) for id in ab_tau_pos_ids])
    _tau_pos = tau_pos[sub_idx]    
      
    ab_times, tau_times = calc_times(ab_tau_pos, _tau_pos)
    
    sub_idx = Vector{Int}()
    ab_idx = Vector{Int}()
    tau_idx = Vector{Int}()
    for (j, (at, tt)) in enumerate(zip(ab_times, tau_times))
        for i in eachindex(tt)
            idx = findfirst(x -> abs(x - tt[i]) <= 0.3, at)
            if idx isa Nothing
                continue
            else
                if length(at[idx:end]) > 1 && length(tt[i:end]) >= min_tau_scans
                    push!(sub_idx, j)
                    push!(ab_idx, idx)
                    push!(tau_idx, i)
                    break
                end
            end
        end
    end
    
    _ab_data_ts = [d[t:end] for (d, t) in zip(ab_tau_pos[sub_idx], ab_idx)]
    _tau_data_ts = [d[t:end] for (d, t) in zip(_tau_pos[sub_idx], tau_idx)]
    ab = ADNIDataset(length(_ab_data_ts), _ab_data_ts, ab_data.rois)
    tau = ADNIDataset(length(_tau_data_ts), _tau_data_ts, tau_data.rois)

    @assert allequal(get_id.(ab) .== get_id.(tau))
    return ab, tau
end

function get_time_idx(d::Vector{Vector{Float64}}, ts::Vector{Vector{Float64}})
    [findall(x -> x ∈ a, t) for (a, t) in zip(d, ts)]
end

function vectorise(d::Vector{Matrix{Float64}})
    reduce(vcat, vec.(d))
end
end