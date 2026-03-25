using DataFrames, CSV
using DrWatson
using Connectomes
using ADNIDatasets
using Dates
using DelimitedFiles
using Statistics

#------------------------------------------------------------------------------------
# ADNI info
#------------------------------------------------------------------------------------
_ab_data_df =  CSV.read(datadir("ADNI/2025/UCBERKELEY_AMY_6MM_28Jul2025.csv"), DataFrame)
_tau_data_df = CSV.read(datadir("ADNI/2025/UCBERKELEY_TAU_6MM_28Jul2025-Ab-tau-Status.csv"), DataFrame) 

demo = CSV.read(datadir("ADNI/2025/demographics.csv"), DataFrame)
diagnostics = CSV.read(datadir("ADNI/2025/ADNIMERGE_16Sep2025.csv"), DataFrame);

df = DataFrame(Group=String[], Age=Float64[], Gender=Float64[], Education=Float64[], CN = Float64[], MCI = Float64[], AD = Float64[], CL= Float64[], CL_std=Float64[])
#-------------------------------------------------------------------------------
# AB pos 
#-------------------------------------------------------------------------------
groups = ["inference-subs", "coloc-subs"]
for g in groups
        id_df = CSV.read(projectdir("data/ADNI/" * g * ".csv"), DataFrame)
        IDs = id_df.sub_id

        n = length(IDs)
        _dmdf = filter(x -> x.RID ∈ IDs, demo)
        idx = [findfirst(isequal(id), _dmdf.RID) for id in IDs]
        posdf = _dmdf[idx,:]

        # Dataframe 
        #-------------------------------------------------------------------------------
        # Tau Pos
        posdf3 = filter(x -> x.RID ∈ IDs, posdf)
        idx = [findfirst(isequal(id), posdf3.RID) for id in IDs]
        posdfinit = posdf3[idx, :]
        
        diag_idx = filter(x -> !isnothing(x), [findfirst(isequal(id), diagnostics.RID) for id in posdfinit.RID])

        pos_diag_df = diagnostics[diag_idx, :]
        pdx_taupos = [count(==(pdx), pos_diag_df.DX_bl) for pdx in ["CN",  "SMC", "EMCI", "LMCI", "AD"]]
        percent_pdx_taupos = pdx_taupos ./ size(pos_diag_df, 1)

        Group = g
        age = mean(year.(posdf.VISDATE) .- posdf.PTDOBYY)
        Gender = count(==(2), posdf.PTGENDER) ./ length(posdf.PTGENDER)
        Education = mean(posdf.PTEDUCAT)
        # amylloid status
        amy_pos = filter(x -> x.RID ∈ IDs, _ab_data_df)
        amy_pos_init_idx = [findfirst(isequal(id), amy_pos.RID) for id in IDs]

        pos_centiloids = mean(amy_pos[amy_pos_init_idx, :].CENTILOIDS)
        pos_centiloids_st = std(amy_pos[amy_pos_init_idx, :].CENTILOIDS)

        push!(df, (Group, age, Gender, Education, 
                sum(percent_pdx_taupos[1:2]), sum(percent_pdx_taupos[3:4]), percent_pdx_taupos[5], pos_centiloids, pos_centiloids_st))
end
df

using PrettyTables  

formatter = (v, i, j) -> round(v, digits = 2);

pretty_table(df; formatters = ft_printf("%5.3f"), backend=Val(:latex))