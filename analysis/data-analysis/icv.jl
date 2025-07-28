using CSV, DataFrames, ADNIDatasets
using DrWatson
using ATNModelling.ConnectomeUtils: get_connectome, get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise, make_ucsf_df,rename_ucsf_df, add_icv, make_ucsf_name, make_dkt_name
using ATNModelling.InferenceModels: fit_model, ensemble_atn_harmonised, ensemble_atn_harmonised_individual

dktnames = get_parcellation() |> get_cortex |> get_dkt_names
# Amyloid data 
_ab_data_df =  CSV.read(datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/UCBERKELEY_TAU_6MM_29Nov2024-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df)
tau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=3)

ucsf_df = CSV.read(datadir("ADNI/2025/adni-ucsf-crosssec-mri.csv"), DataFrame)
data_dict_df = CSV.read(datadir("ADNI/adni-data-dictionary.csv"), DataFrame)
ucsf_data_dict = filter(x -> x.CRFNAME == "Cross-Sectional FreeSurfer (7.x)", data_dict_df)
atr_df = make_ucsf_df(ucsf_df, ucsf_data_dict, dktnames)
tau_atr_df = add_icv(tau_pos_df, atr_df; dt_threshold=180)

tau_icv_df = filter(x -> x.Has_ICV, tau_atr_df)
_, ucsf_dkt_names = make_ucsf_name(ucsf_data_dict, dktnames)
vol_names = filter(x -> startswith(x, "CorticalVolume") || startswith(x, "SubcorticalVolume"), ucsf_dkt_names)

# tautau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, tau_icv_df)
tau_data = ADNIDataset(tau_icv_df, dktnames; min_scans=3)

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

get_id.(fbb_tau)

fbb_icv = [filter(x -> x.RID == rid, tau_icv_df).ICV for rid in get_id.(fbb_tau)]
allequal(length.(fbb_icv) .== size.(calc_suvr.(fbb_tau), 2))

fbp_icv = [filter(x -> x.RID == rid, tau_icv_df).ICV for rid in get_id.(fbp_tau)]
allequal(length.(fbp_icv) .== size.(calc_suvr.(fbp_tau), 2))