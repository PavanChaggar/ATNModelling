using CSV
using DataFrames
using DrWatson: projectdir, datadir
using Statistics
using ATNModelling.DataUtils: set_ab_status, set_tau_status
using ATNModelling.ParcellationUtils: get_parcellation, get_cortex, get_dkt_names
using Connectomes: connectome_path, Parcellation, get_label

# ------------------------------------------------------------------------------------------
# AB status
# ------------------------------------------------------------------------------------------

_taudata = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024.csv"), DataFrame) 
taudata = filter(x -> x.TRACER == "FTP" && x.qc_flag==2, _taudata)
_abdata =  CSV.read(datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv"), DataFrame)
abdata = filter(x -> x.qc_flag==2, _abdata)

tau_data_ab_status = set_ab_status(abdata, taudata) 

CSV.write(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-AB-Status.csv"), tau_data_ab_status)

# ------------------------------------------------------------------------------------------
# Tau status
# ------------------------------------------------------------------------------------------
parc = Parcellation(connectome_path())
cortex = filter(x -> x.Lobe != "subcortex", parc)

mtl_regions = ["entorhinal", "Left-Amygdala", "Right-Amygdala"] 
mtl = findall(x -> x ∈ mtl_regions, get_label.(cortex)) 
neo_regions = ["inferiortemporal", "middletemporal"] 
neo = findall(x -> x ∈ neo_regions, get_label.(cortex))

dktnames = get_parcellation() |> get_cortex |> get_dkt_names
df_names = uppercase.(dktnames) .* "_SUVR" 

mtl_cutoff = 1.375 # Medial temporal lobe cutoff
neo_cutoff = 1.395 # Neocortical cutoff

tau_status = set_tau_status(tau_data_ab_status, df_names, mtl, neo, mtl_cutoff, neo_cutoff)

CSV.write(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), tau_status) 