using CSV, DataFrames, ADNIDatasets
using DrWatson
using ADNIDatasets.DataConverter
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise
using ATNModelling.InferenceModels: fit_model, ensemble_atn_harmonised, ensemble_atn_harmonised_individual

dktnames = get_parcellation() |> get_cortex |> get_dkt_names
# Amyloid data 
_ab_data_df =  CSV.read(datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df)
tau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, tau_data_df);

atr_df = CSV.read(datadir("ADNI/ucsf-FS-smri.csv"), DataFrame) 

tau_atr_df = DataConverter.add_icv(tau_data_df, atr_df; dt_threshold=180)

tau_icv_df = filter(x -> x.Has_ICV, tau_atr_df)

# tautau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, tau_icv_df)
tau_data = ADNIDataset(tau_icv_df, dktnames; min_scans=3, include_icv=true)
get_icv.(tau_data)
ADNIDatasets.get_dates.(tau_data)
tracer="FBB"
# fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

fbb, fbb_tau = align_data(fbb_data, tau_data; min_tau_scans=3)

tracer="FBP"
# fbp_u0, fbp_ui = load_ab_params(tracer=tracer)
fbp_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
fbp_data = ADNIDataset(fbp_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")
fbp, fbp_tau = align_data(fbp_data, tau_data; min_tau_scans=3)

