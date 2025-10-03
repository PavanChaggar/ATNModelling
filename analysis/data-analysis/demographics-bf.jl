using CSV
using DataFrames
using DrWatson: projectdir, datadir
using Statistics
using ATNModelling.ConnectomeUtils: get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise
include(projectdir("bf-data.jl"))

data_path = datadir("bf-data/bf-data-ab-tau-summary.csv");

data_df = CSV.read(data_path, DataFrame)

dktnames = get_parcellation() |> get_cortex |> get_dkt_names
data = BFDataset(data_df, dktnames; min_scans=2, tracer=:ab)
sid = "BF" .* string.(get_id.(data))

_idx = [findfirst(x -> x == _sid, data_df.sid) for _sid in sid]
abpos = data_df[_idx,:]
abpos.age |> mean
count(==(0), abpos.gender_baseline_variable) / 812
filter(!ismissing, abpos.education_level_years_baseline_variable) |> mean
[count(==(x), abpos.diagnosis_baseline_variable) for x in ["AD", "SCD", "MCI", "Normal"]] / 812

st = ["NormalNot_determined", 
"SCDNot_determined",
"MCINot_determined",
"NormalAD",
"SCDAD",
"MCIAD",
]

st = ["Normal", "SCD", "MCI", "Dementia"]
unique(abpos.diagnosis_baseline_variable)
status = abpos.diagnosis_baseline_variable .* abpos.underlying_etiology_text_baseline_variable
st_count = [count(==(x), status) for x in st] / 813
sum(st_count)
sum(st_count)

abneg_tau = filter(x -> x.ab_status == 0, data_df)
data = BFDataset(abneg_tau, dktnames; min_scans=1, tracer=:tau)
sid = "BF" .* string.(get_id.(data))
_idx = [findfirst(x -> x == _sid, data_df.sid) for _sid in sid]
abpos = data_df[_idx,:]
abpos.age |> mean
count(==(0), abpos.gender_baseline_variable) / 938
filter(!ismissing, abpos.education_level_years_baseline_variable) |> mean
dropmissing!(abpos, ["diagnosis_baseline_variable","underlying_etiology_text_baseline_variable"])
status = abpos.diagnosis_baseline_variable .* abpos.underlying_etiology_text_baseline_variable
st_count = [count(==(x), status) for x in st] / 877


ab_data_df = filter(x -> x.ab_status == 1, data_df)
ab_data = BFDataset(ab_data_df, dktnames; min_scans=3, tracer=:ab)

# Tau data 
tau_pos_df = filter(x ->  x.MTL_Status == 1 || x.NEO_Status == 1, ab_data_df);
tau_data = BFDataset(tau_pos_df, dktnames; min_scans=3, tracer=:tau)

ab, tau = align_data(ab_data, tau_data)

sid = "BF" .* string.(get_id.(tau))
_idx = [findfirst(x -> x == _sid, data_df.sid) for _sid in sid]
abpos = data_df[_idx,:]
abpos.age |> mean
count(==(0), abpos.gender_baseline_variable) / 38
filter(!ismissing, abpos.education_level_years_baseline_variable) |> mean
[count(==(x), abpos.underlying_etiology_text_baseline_variable) for x in ["AD", "SCD", "MCI", "Normal"]] / 48
dropmissing!(abpos, ["diagnosis_baseline_variable","underlying_etiology_text_baseline_variable"])
status = abpos.diagnosis_baseline_variable .* abpos.underlying_etiology_text_baseline_variable
st_count = [count(==(x), status) for x in st] / 48
sum(st_count[2:3])
st = ["NormalNot_determined", 
"SCDNot_determined",
"MCINot_determined",
"NormalAD",
"SCDAD",
"MCIAD",
]

st = ["Normal", "SCD", "MCI", "AD"]
unique(abpos.diagnosis_baseline_variable)
status = abpos.cognitive_status_baseline_variable .* abpos.underlying_etiology_text_baseline_variable
status = abpos.cognitive_status_baseline_variable
st_count = [count(==(x), status) for x in st] / 48
abpos.CL_fnc_ber_com_composite |> mean
abpos.CL_fnc_ber_com_composite |> std
