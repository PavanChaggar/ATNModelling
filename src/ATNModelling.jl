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
using SciMLBase
using Turing
using LinearAlgebra
using ADTypes

include("data-utils.jl")
include("connectome-utils.jl")
include("simulation-utils.jl")
include("inference-models.jl")

using .DataUtils
using .ConnectomeUtils
using .SimulationUtils
using .InferenceModels

end
