using CSV
using DataFrames
using DrWatson: projectdir, datadir
using Statistics
using ATNModelling.ConnectomeUtils: get_parcellation, get_cortex, get_dkt_names
using ATNModelling.DataUtils: align_data, normalise!, get_time_idx, vectorise
include(projectdir("bf-data.jl"))

data_path = datadir("bf-data/bf-data-ab-tau-summary.csv");

data_df = CSV.read(data_path, DataFrame)

df = DataFrame(Group=String[], Age=Float64[], Gender=Float64[], Education=Float64[], CN = Float64[], MCI = Float64[], AD = Float64[], CL= Float64[], CL_std=Float64[])
Group = "inference"
st = ["Normal", "SCD", "MCI", "AD"]
for Group in ["inference-subs", "coloc-subs"]
    id_df = CSV.read(projectdir("data/bf-data/" * Group * ".csv"), DataFrame)

    sid = "BF" .* string.(id_df.sub_id)
    n =length(sid)
    println(n)
    _idx = [findfirst(x -> x == _sid, data_df.sid) for _sid in sid]
    abpos = data_df[_idx,:]
    age = abpos.age |> mean
    Gender = count(==(0), abpos.gender_baseline_variable) / n

    Education = filter(!ismissing, abpos.education_level_years_baseline_variable) |> mean
    dropmissing!(abpos, :diagnosis_baseline_variable)
    status = abpos.diagnosis_baseline_variable
    println(unique(status))
    st_count = [count(==(x), status) for x in st] / n
    pos_centiloids = abpos.CL_fnc_ber_com_composite |> mean
    pos_centiloids_st = abpos.CL_fnc_ber_com_composite |> std

    push!(df, (Group, age, Gender, Education, 
                    st_count[1], sum(st_count[2:3]), st_count[4], pos_centiloids, pos_centiloids_st))
end
df


using PrettyTables  

formatter = (v, i, j) -> round(v, digits = 2);

pretty_table(df; formatters = ft_printf("%5.3f"), backend=Val(:latex))