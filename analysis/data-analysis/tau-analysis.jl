using ATNModelling
using ATNModelling.ConnectomeUtils: get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: get_dkt_moments
using ATNModelling.SimulationUtils: conc
using ADNIDatasets: ADNIDataset, get_initial_conditions
using DrWatson: datadir, projectdir
using CSV, DataFrames
using DelimitedFiles: writedlm
using Connectomes: get_lobe

taudata = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame);
taudata
allequal(taudata.TRACER)
abneg_tau = filter(x -> x.AB_Status == 0, taudata)

cortex = get_parcellation() |> get_cortex
dktnames = get_dkt_names(cortex)

neg_tau_data = ADNIDataset(abneg_tau, dktnames; min_scans=1)
# N = 582

neg_subdata = get_initial_conditions.(neg_tau_data)

for i in 1:72
    x_data = [n[i] for n in neg_subdata]
    writedlm(projectdir("py-analysis/roi-data/data-$i.csv"), x_data)
end

writedlm(projectdir("py-analysis/temporal-rois.csv"), findall(x -> get_lobe(x) == "temporal", cortex))

pypart = CSV.read(projectdir("output/analysis-derivatives/tau-derivatives/pypart.csv"), DataFrame)

tau_params = CSV.read(datadir("adni-derivatives/tau-params.csv"), DataFrame)
v0 = tau_params.v0
vi = tau_params.vi

partbase, partpath = get_dkt_moments(pypart,  pypart.region)
part_vi = quantile.(partpath, .99)
_part = deepcopy(v0)
_part[Int.(pypart.region)] .= part_vi


d = [_part[1:36]; _part[37:end]]
part_increase = d .- v0
part_sym_increase = (part_increase[1:36] .+ part_increase[37:end]) ./ 2
sympart = v0 .+ [part_sym_increase; part_sym_increase]

writedlm(projectdir("output/analysis-derivatives/tau-derivatives/pypart-sym.csv"), sympart)

# Cutoffs 
using GaussianMixtures, Distributions
data = ADNIDataset(taudata, dktnames; qc=true)
d = reduce(hcat, get_initial_conditions.(data))

_ics = get_initial_conditions.(data)
ics = _ics[findall(x -> x < 3, reduce(hcat, _ics)[4,:])]
d = reduce(hcat, ics)
ms = Vector{Float64}()
stds = Vector{Float64}()
ws = Vector{Float64}()
ms2 = Vector{Float64}()
stds2 = Vector{Float64}()
ws2 = Vector{Float64}()
gmms = Vector{GMM}()

for i in 1:72
    gmm = GMM(2, reshape(d[i, :], 1021, 1))
    μ = means(gmm)
    Σ = covars(gmm)
    w = weights(gmm)
    idx1 = argmin(μ)
    idx2 = argmax(μ)
    push!(gmms, gmm)
    push!(ms, μ[idx1])
    push!(stds, sqrt(Σ[idx1]))
    push!(ws, weights(gmm)[idx1])
    push!(ms2, μ[idx2])
    push!(stds2, sqrt(Σ[idx2]))
    push!(ws2, weights(gmm)[idx2])
end

# cutoffs = quantile.(Normal.(ms, stds), 0.75)
mean(conc.(cutoffs, v0, vi))
cutoffs = ms .+ 2 .* stds
mean(conc.(cutoffs, v0, vi))
writedlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-2std.csv"), cutoffs)
cutoffs = ms .+ 1.5 .* stds
mean(conc.(cutoffs, v0, vi))
writedlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-1std.csv"), cutoffs)

threshold = Vector{Float64}()
xs = reshape(collect(1.:0.001:3.5), 2501, 1)
for gmm in gmms
    gmmp = gmmposterior(gmm, xs)
    idx = argmin(abs.(gmmp[1][:,1] .- 0.5))
    push!(threshold, xs[idx])
end
writedlm(projectdir("output/analysis-derivatives/tau-derivatives/tau-cutoffs-halfprob.csv"), threshold)

include(projectdir("bf-data.jl"))

data_path = datadir("bf-data/bf-data-ab-tau-summary.csv");
data_df = CSV.read(data_path, DataFrame)
data = BFDataset(data_df, dktnames; min_scans=1, tracer=:tau)

ics = get_initial_conditions.(data)
# ics = _ics[findall(x -> x < 3, reduce(hcat, _ics)[4,:])]
d = reduce(hcat, ics)
ms = Vector{Float64}()
stds = Vector{Float64}()
ws = Vector{Float64}()
ms2 = Vector{Float64}()
stds2 = Vector{Float64}()
ws2 = Vector{Float64}()
gmms = Vector{GMM}()

for i in 1:72
    gmm = GMM(2, reshape(d[i, :], 1598, 1))
    μ = means(gmm)
    Σ = covars(gmm)
    w = weights(gmm)
    idx1 = argmin(μ)
    idx2 = argmax(μ)
    push!(gmms, gmm)
    push!(ms, μ[idx1])
    push!(stds, sqrt(Σ[idx1]))
    push!(ws, weights(gmm)[idx1])
    push!(ms2, μ[idx2])
    push!(stds2, sqrt(Σ[idx2]))
    push!(ws2, weights(gmm)[idx2])
end

v0, vi, part = load_tau_params(tracer="RO")
cutoffs = ms .+ 2 .* stds
mean(conc.(cutoffs, v0, vi))
writedlm(projectdir("output/analysis-derivatives/bf/tau-derivatives/tau-cutoffs-2std-bf.csv"), cutoffs)
cutoffs = ms .+ 1.5 .* stds
mean(conc.(cutoffs, v0, vi))
writedlm(projectdir("output/analysis-derivatives/bf/tau-derivatives/tau-cutoffs-1std-bf.csv"), cutoffs)

threshold = Vector{Float64}()
xs = reshape(collect(1.:0.001:4.5), 3501, 1)
for gmm in gmms
    gmmp = gmmposterior(gmm, xs)
    idx = argmin(abs.(gmmp[1][:,1] .- 0.5))
    push!(threshold, xs[idx])
end
writedlm(projectdir("output/analysis-derivatives/bf/tau-derivatives/tau-cutoffs-halfprob-bf.csv"), threshold)


fg(x, μ, σ) = exp.(.-(x .- μ) .^ 2 ./ (2σ^2)) ./ (σ * √(2π))

function plot_density!(μ, Σ, weight; color=:blue, label="")
    d = Normal(μ, Σ)
    x = LinRange(quantile(d, .00001),quantile(d, .99999), 200)
    lines!(x, weight .* fg(x, μ, Σ); color = color, label=label)
    band!(x, fill(0, length(x)), weight .* fg(x, μ, Σ); color = (color, 0.1), label=label)
end


begin
    f = Figure()
    ax = Axis(f[1,1])
    node = 4
    hist!(reduce(hcat, ics)[node,:], normalization=:pdf, bins=20)
    # plot!( Normal.(ms, stds)[node])
    # plot!( Normal.(ms2, stds2)[node])
    plot_density!(ms[node], stds[node], ws[node])
    plot_density!(ms2[node], stds2[node], ws2[node])
    vlines!(cutoffs[node])
    vlines!(threshold[node])
    f
end
