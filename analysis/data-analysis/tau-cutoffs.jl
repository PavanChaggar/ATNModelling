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

abneg_tau = filter(x -> x.AB_Status == 0, taudata)

cortex = get_parcellation() |> get_cortex
dktnames = get_dkt_names(cortex)

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