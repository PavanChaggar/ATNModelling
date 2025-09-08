using DataFrames
using CSV
using Connectomes
using ADNIDatasets
using Dates
using DrWatson
using DelimitedFiles
parc = Parcellation(Connectomes.connectome_path())
cortex = filter(x -> get_lobe(x) != "subcortex", parc);

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"]
mtl = findall(x -> get_label(x) ∈ mtl_regions, cortex)
neo_regions = ["inferiortemporal", "middletemporal"]
neo = findall(x -> get_label(x) ∈ neo_regions, cortex)

dktdict = Connectomes.node2FS()
dktnames = [dktdict[i] for i in get_node_id.(cortex)]

ctx_idx = findall( x-> !contains(x, "-"), get_label.(cortex))
_vol_names = replace.(get_label.(cortex), "-" => "_")

prefix_dict = Dict("cortical" => "aparc_grayvol_", "subcortical" => "aseg_vol_" )
hemisphere_dict = Dict("left" => "_L", "right" => "_R")
prefixes = [prefix_dict[x] for x in Connectomes.get_cortex.(cortex)]
suffixes = [hemisphere_dict[x] for x in get_hemisphere.(cortex)]
vol_names = prefixes .* _vol_names
vol_names[ctx_idx] .= vol_names[ctx_idx] .* suffixes[ctx_idx]
vol_names

ab_names = "fnc_sr_mr_fs_" .* dktnames
tau_names = "tnic_sr_mr_fs_" .* dktnames

ab_names = readdlm(datadir("bf-derivatives/ab-names.csv")) |> vec
tau_names = readdlm(datadir("bf-derivatives/tau-names.csv")) |> vec
vol_names = readdlm(datadir("bf-derivatives/vol-names.csv")) |> vec

ab_dict = Dict(zip(dktnames, ab_names))
ab_dict["ab_summary"] = "ab_summary"
ab_dict["fnc_ber_com_composite"] = "fnc_ber_com_composite"
tau_dict = Dict(zip(dktnames, tau_names))
vol_dict = Dict(zip(dktnames, vol_names))
vol_dict["ab_summary"] = "ab_summary_vol"
vol_dict["fnc_ber_com_composite"] = "fnc_ber_com_composite_vol"

(d::Dict)(k) = d[k]

function BFSubject(subid, df, roi_names, tracer)
    sub = filter(x -> x.sid == subid, df)

    if tracer == :ab
        subdate = sub[:,"flute_pet_date"] 
        subsuvr = sub[:, ab_dict.(roi_names)] |> dropmissing |> disallowmissing |> Array
    elseif tracer == :tau
        subdate = sub[:,"tau_pet_date"] 
        subsuvr = sub[:, tau_dict.(roi_names)] |> dropmissing |> disallowmissing |> Array
    end

    subvol = sub[:, vol_dict.(roi_names)] |> dropmissing |> disallowmissing |> Array

    n_scans = length(subdate)
    _subid = parse( Int, subid[3:end])
    if n_scans == size(subsuvr,1) == size(subvol,1)
        return ADNISubject(
            _subid, 
            n_scans, 
            subdate, 
            [ADNIScanData(subdate[i], subsuvr[i,:], subvol[i, :], 1.0, 1.0)
            for i in 1:n_scans]
        )
    end
end


function BFDataset(df, roi_names; min_scans=1, max_scans=Inf, tracer=:ab)

    subjects = unique(df.sid)
    n_scans = [count(==(sub), df.sid) for sub in subjects]
    multi_subs = subjects[findall(x -> min_scans <= x <= max_scans, n_scans)]

    bfsubjects = Vector{ADNISubject}()
    for sub in multi_subs
        _sub = BFSubject(sub, df, roi_names, tracer)
        if _sub isa ADNISubject
            push!(bfsubjects, _sub)
        end
    end
    ADNIDataset(length(bfsubjects), bfsubjects, roi_names)
end

function get_dkt_moments_biofinder(gmm_moments, dktnames)
    μ_1 = Vector{Float64}()
    μ_2 = Vector{Float64}()

    σ_1 = Vector{Float64}()
    σ_2 = Vector{Float64}()
    
    for name in dktnames[1:end]
        roi = filter( x -> x.ROI == name, gmm_moments)
        push!(μ_1, roi.mu_low_distribution[1])
        push!(μ_2, roi.mu_high_distribution[1])
        push!(σ_1, roi.sigma_low_distribution[1])
        push!(σ_2, roi.sigma_high_distribution[1])
    end
    Normal.(μ_1, σ_1), Normal.(μ_2, σ_2)
end