using ATNModelling
using ATNModelling.ParcellationUtils: get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: get_dkt_moments
using ADNIDatasets: ADNIDataset, get_initial_conditions
using DrWatson: datadir, projectdir
using CSV, DataFrames
using DelimitedFiles: writedlm
using Connectomes: get_lobe

taudata = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame);

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

gmm_moments = CSV.read(datadir("derivatives/component_moments.csv"), DataFrame)
ubase, upath = get_dkt_moments(gmm_moments, dktnames)
v0 = mean.(ubase)
vi = quantile.(upath, .99)

partbase, partpath = get_dkt_moments(pypart,  pypart.region)
part_vi = quantile.(partpath, .99)
_part = deepcopy(v0)
_part[Int.(pypart.region)] .= part_vi

_sympart = mean.(collect((zip(_part[1:36], _part[37:end]))))
sympart = [_sympart; _sympart]

writedlm(projectdir("output/analysis-derivatives/tau-derivatives/pypart-sym.csv"), sympart)
