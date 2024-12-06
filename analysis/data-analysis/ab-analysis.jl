using ATNModelling
using ATNModelling.DataUtils: baseline_difference, split_data, fit_second_order_polynomial, 
                   find_amyloid_time, find_regional_params
using ATNModelling.ConnectomeUtils: get_parcellation, get_cortex, get_dkt_names
using ADNIDatasets: ADNIDataset, get_id, get_initial_conditions

using CSV, DataFrames
using DrWatson: projectdir, datadir
using DelimitedFiles: writedlm
using Polynomials: coeffs

data_path = datadir("ADNI/UCBERKELEY_AMY_6MM_29Nov2024.csv");

data_df = CSV.read(data_path, DataFrame)
fbp_data = filter(x -> x.TRACER == "FBP", data_df)
dropmissing!(fbp_data, :AMYLOID_STATUS_COMPOSITE_REF)

abpos_df = filter(x -> x["AMYLOID_STATUS_COMPOSITE_REF"] ∈ [0, 1], fbp_data)

dktnames = get_parcellation() |> get_cortex |> get_dkt_names
push!(dktnames, "SUMMARY")

data = ADNIDataset(abpos_df, dktnames; min_scans=2, reference_region="COMPOSITE_REF", qc=false)

ab_vf = baseline_difference(data, 73)

CSV.write(projectdir("output/analysis-derivatives/ab-derivatives/ab-vector-field.csv"), ab_vf)

start = 0.65
step  = 0.05
stop  = 1.15

bin_df = split_data(ab_vf.ab_suvr, ab_vf.ab_diff, start, step, stop)
CSV.write(projectdir("output/analysis-derivatives/ab-derivatives/ab-binned-vector-field.csv"), bin_df)

f, roots = fit_second_order_polynomial(bin_df.ab_bin, bin_df.ab_bin_diffs)
writedlm(projectdir("output/analysis-derivatives/ab-derivatives/ab-polynomial-coeffs.csv"), coeffs(f))
# f = 0.16844315367056056 + 0.4146967939561749*x - 0.2330383236437777*x^2 
# roots = [0.627346338525677, 1.1521755335509531]

#-----------------------------------------------------------------------
# Ab integration
#-----------------------------------------------------------------------
xt(t) = (32.298 + 1.152 * exp(0.122 * t))/(51.483 + exp(0.122 * t))
xt(t, p) = (32.298 + 1.152 * exp(0.122 * t))/(51.483 + exp(0.122 * t)) - p

t_df = find_amyloid_time(xt, data);
CSV.write(projectdir("output/analysis-derivatives/ab-derivatives/ab-times.csv"), t_df)

t_df = CSV.read(projectdir("output/analysis-derivatives/ab-derivatives/ab-times.csv"), DataFrame)

params = find_regional_params(data, t_df);

u0 = [params[i].param[4] for i in 1:72]
diff = [params[i].param[1] for i in 1:72]
ui = [diff[i] + u0[i] for i in 1:72]
α =  [params[i].param[2] for i in 1:72]
t50 =  [params[i].param[3] for i in 1:72]

dfparams = DataFrame(u0 = u0, alpha = α, t50 = t50, ui = ui, diff = diff)

CSV.write(projectdir("output/analysis-derivatives/ab-derivatives/ab-params.csv"), dfparams)