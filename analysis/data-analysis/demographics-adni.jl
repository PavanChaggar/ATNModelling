using DataFrames, CSV
using DrWatson
using Connectomes
using ADNIDatasets
using Dates
using DelimitedFiles
using Statistics

#------------------------------------------------------------------------------------
# A+T+
#------------------------------------------------------------------------------------
# Amyloid data 
_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 

tau_data_df = filter(x -> x.qc_flag==2 && x.AB_Status == 1, _tau_data_df);
tau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, tau_data_df);
tau_data = ADNIDataset(tau_pos_df, dktnames; min_scans=3)

ucsf_df = CSV.read(datadir("ADNI/2025/adni-ucsf-crosssec-mri.csv"), DataFrame)
data_dict_df = CSV.read(datadir("ADNI/adni-data-dictionary.csv"), DataFrame)
ucsf_data_dict = filter(x -> x.CRFNAME == "Cross-Sectional FreeSurfer (7.x)", data_dict_df)
atr_df = make_ucsf_df(ucsf_df, ucsf_data_dict, dktnames)
filter!(x -> x.OVERALLQC != "Fail", atr_df)
tau_atr_df = add_icv(tau_pos_df, atr_df; dt_threshold=180)
tau_icv_df = filter(x -> x.Has_ICV, tau_atr_df)

tau_data = ADNIDataset(tau_icv_df, dktnames; min_scans=3)
# --------------------------------------------------------------------------------
# Load fbb + fbp data
# --------------------------------------------------------------------------------
tracer="FBB"
fbb_u0, fbb_ui = load_ab_params(tracer=tracer)
fbb_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
fbb_data = ADNIDataset(fbb_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

fbb, fbb_tau = align_data(fbb_data, tau_data; min_tau_scans=3)

tracer="FBP"
fbp_u0, fbp_ui = load_ab_params(tracer=tracer)
fbp_data_df = filter(x -> x.qc_flag==2 && x.TRACER == tracer, _ab_data_df);
fbp_data = ADNIDataset(fbp_data_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF")

fbp, fbp_tau = align_data(fbp_data, tau_data; min_tau_scans=3)

#------------------------------------------------------------------------------------
# ADNI info
#------------------------------------------------------------------------------------
demo = CSV.read(datadir("ADNI/2025/demographics.csv"), DataFrame)
diagnostics = CSV.read(datadir("ADNI/2025/ADNIMERGE_16Sep2025.csv"), DataFrame);

#-------------------------------------------------------------------------------
# AB pos 
#-------------------------------------------------------------------------------
IDs = [get_id(fbb); get_id(fbp)]

_dmdf = reduce(vcat, [filter(x -> x.RID == id, demo) for id in IDs])
idx = [findfirst(isequal(id), _dmdf.RID) for id in IDs]
posdf = _dmdf[idx,:]

# Dataframe 
df = DataFrame(Group=String[], Age=Float64[], Gender=Float64[], Education=Float64[], CN = Float64[], MCI = Float64[], AD = Float64[])
#-------------------------------------------------------------------------------
# Tau Pos
posdf3 = filter(x -> x.RID ∈ IDs, posdf)
idx = [findfirst(isequal(id), posdf3.RID) for id in IDs]
posdfinit = posdf3[idx, :]

diag_idx = [findfirst(isequal(id), diagnostics.RID) for id in posdfinit.RID]
pos_diag_df = diagnostics[diag_idx, :]
pdx_taupos = [count(==(pdx), pos_diag_df.DX_bl) for pdx in ["CN",  "SMC", "EMCI", "LMCI", "AD"]]
percent_pdx_taupos = pdx_taupos ./ 34

Group = "tau +"
ages = [[fbb_tau.SubjectData[i].scan_dates[1] for i in 1:18]; 
        [fbp_tau.SubjectData[i].scan_dates[1] for i in 1:16]]
age = mean(year.(ages) .- (posdf.PTDOBYY))
Gender = count(==(2), posdf.PTGENDER) ./ length(posdf.PTGENDER)
Education = mean(posdf.PTEDUCAT)


# amylloid status
amy_pos = filter(x -> x.RID ∈ IDs, _ab_data_df)
amy_pos_init_idx = [findfirst(isequal(id), amy_pos.RID) for id in IDs]

pos_centiloids = mean(amy_pos[amy_pos_init_idx, :].CENTILOIDS)
pos_centiloids_st = std(amy_pos[amy_pos_init_idx, :].CENTILOIDS)


push!(df, (Group, age, Gender, Education, 
          sum(percent_pdx_taupos[1:2]), sum(percent_pdx_taupos[3:4]), percent_pdx_taupos[5]))
#-------------------------------------------------------------------------------
# Tau neg
#-------------------------------------------------------------------------------
ab_sub_data_path = projectdir("data/UCBERKELEY_AMY_6MM_24Jun2023.csv")
alldf = CSV.read(ab_sub_data_path, DataFrame)
dropmissing!(alldf)

pos_ab_df = filter(x -> x["AMYLOID_STATUS_COMPOSITE_REF"] == 1, alldf)

ab_data = ADNIDataset(pos_ab_df, dktnames; min_scans=1, reference_region="COMPOSITE_REF", qc=false)

tau_data = ADNIDataset(pos_tau_df, dktnames; min_scans=1, qc=false)

_, tau_neg = tau_positivity(tau_data, mtl, neo, mtl_cutoff, neo_cutoff)

neg_ids = get_id.(tau_neg)

ab, tau = align_data(ab_data, tau_neg)

data = tau

IDs = get_id(data)

_dmdf = reduce(vcat, [filter(x -> x.RID == id, demo) for id in IDs])
idx = [findfirst(isequal(id), _dmdf.RID) for id in IDs]
negdf = _dmdf[idx,:]

neg_diag_idx = [findfirst(isequal(id), diagnostics.RID) for id in negdf.RID]
neg_diag_df = diagnostics[neg_diag_idx, :]
pdx_tauneg = [count(==(pdx), neg_diag_df.DX_bl) for pdx in ["CN",  "SMC", "EMCI", "LMCI", "AD"]]
percent_pdx_tauneg = pdx_tauneg ./ sum(pdx_tauneg)

Group = "tau -"
age = mean(year.([data.SubjectData[i].scan_dates[1] for i in 1:71]) .- (negdf.PTDOBYY))
Gender = count(==(2), negdf.PTGENDER) ./ length(negdf.PTGENDER)
Education = mean(negdf.PTEDUCAT)
push!(df, (Group, age, Gender, Education, sum(percent_pdx_tauneg[1:2]), sum(percent_pdx_tauneg[3:4]), percent_pdx_tauneg[5]))

#-------------------------------------------------------------------------------
# AB neg 
#-------------------------------------------------------------------------------
tau_sub_data_path = projectdir("data/UCBERKELEYAV1451_8mm_02_17_23_AB_Status.csv")
tau_alldf = CSV.read(tau_sub_data_path, DataFrame)
tau_names = dktnames[1:70]
# push!(tau_names, "META_TEMPORAL")

tau_data = ADNIDataset(tau_alldf, tau_names; min_scans=1, qc=false)
n_tau = length(tau_data)

neg_tau_df = filter(x -> x.AB_Status == 0, tau_alldf)

tau_neg = ADNIDataset(neg_tau_df, tau_names; min_scans=1, qc=false)

negIDs = get_id(tau_neg)

_dmdfneg = reduce(vcat, [filter(x -> x.RID == id, demo) for id in negIDs])
idxneg = [findfirst(isequal(id), _dmdfneg.RID) for id in negIDs]
dmdfneg = _dmdfneg[idxneg,:]

Group = "Ab-t+"
age = mean(year.([tau_neg.SubjectData[i].scan_dates[1] for i in 1:length(tau_neg)]) .- (dmdfneg.PTDOBYY))
Gender = count(==(2), dmdfneg.PTGENDER) ./ length(dmdfneg.PTGENDER)
Education = mean(dmdfneg.PTEDUCAT)

negdf3 = filter(x -> x.RID ∈ negIDs, dmdfneg)
idx = [findfirst(isequal(id), negdf3.RID) for id in negIDs]
abnegdfinit = negdf3[idx, :]

ab_diag_idx = [findfirst(isequal(id), diagnostics.RID) for id in abnegdfinit.RID]
ab_diag_df = diagnostics[ab_diag_idx, :]
pdx_ab = [count(==(pdx), ab_diag_df.DX_bl) for pdx in ["CN",  "SMC", "EMCI", "LMCI", "AD"]]
percent_pdx_ab = pdx_ab ./ sum(pdx_ab)

push!(df, (Group, age, Gender, Education, sum(percent_pdx_ab[1:2]), sum(percent_pdx_ab[3:4]), percent_pdx_ab[5]))

#-------------------------------------------------------------------------------
# Ab data
#-------------------------------------------------------------------------------
ab_sub_data_path = projectdir("data/UCBERKELEY_AMY_6MM_24Jun2023.csv")
alldf = CSV.read(ab_sub_data_path, DataFrame)
dropmissing!(alldf)

ab_data = ADNIDataset(alldf, dktnames; min_scans=2, reference_region="COMPOSITE_REF", qc=false)

ids = get_id(ab_data)
_dmdfneg = reduce(vcat, [filter(x -> x.RID == id, demo) for id in ids])
idxneg = [findfirst(isequal(id), _dmdfneg.RID) for id in ids]
dmdfneg = _dmdfneg[idxneg,:]

Group = "Ab"
age = mean(year.([ab_data.SubjectData[i].scan_dates[1] for i in 1:length(ab_data)]) .- (dmdfneg.PTDOBYY))
Gender = count(==(2), dmdfneg.PTGENDER) ./ length(dmdfneg.PTGENDER)
Education = mean(dmdfneg.PTEDUCAT)

negdf3 = filter(x -> x.RID ∈ ids, dmdfneg)
idx = [findfirst(isequal(id), negdf3.RID) for id in ids]
abnegdfinit = negdf3[idx, :]

ab_diag_idx = [findfirst(isequal(id), diagnostics.RID) for id in abnegdfinit.RID]
ab_diag_df = diagnostics[filter(x -> x != nothing, ab_diag_idx), :]
pdx_ab = [count(==(pdx), ab_diag_df.DX_bl) for pdx in ["CN",  "SMC", "EMCI", "LMCI", "AD"]]
percent_pdx_ab = pdx_ab ./ sum(pdx_ab)

push!(df, (Group, age, Gender, Education, sum(percent_pdx_ab[1:2]), sum(percent_pdx_ab[3:4]), percent_pdx_ab[5]))

using PrettyTables
formatter = (v, i, j) -> round(v, digits = 3);
(df, digits=3)

pretty_table(df; formatters = ft_printf("%5.4f"), backend=Val(:latex))