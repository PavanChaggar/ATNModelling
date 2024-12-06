module ATNModelling

using Connectomes
using ADNIDatasets
using Statistics: mean
using Polynomials
using NonlinearSolve
using DataFrames
using LsqFit
using DelimitedFiles
using DifferentialEquations

include("data-utils.jl")
include("connectome-utils.jl")
include("simulation-utils.jl")

using .DataUtils
using .ConnectomeUtils
using .SimulationUtils

end
