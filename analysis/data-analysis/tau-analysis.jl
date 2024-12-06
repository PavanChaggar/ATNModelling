using ATNModelling
using ATNModelling.ConnectomeUtils: get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: get_dkt_moments
using ADNIDatasets: ADNIDataset, get_initial_conditions
using DrWatson: datadir, projectdir
using CSV, DataFrames
using DelimitedFiles: writedlm
using Connectomes: get_lobe

taudata = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame);
taudata
abneg_tau = filter(x -> x.AB_Status == 0, taudata)

cortex = get_parcellation() |> get_cortex
dktnames = get_dkt_names(cortex)

neg_tau_data = ADNIDataset(abneg_tau, dktnames; min_scans=1)
neg_subdata = get_initial_conditions.(neg_tau_data)

for i in 1:72
    x_data = [n[i] for n in neg_subdata]
    writedlm(projectdir("py-analysis/roi-data/data-$i.csv"), x_data)
end

writedlm(projectdir("py-analysis/temporal-rois.csv"), findall(x -> get_lobe(x) == "temporal", cortex))

pypart = CSV.read(projectdir("output/analysis-derivatives/tau-derivatives/pypart.csv"), DataFrame)

tau_params = CSV.read(datadir("derivatives/tau-params.csv"), DataFrame)
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
