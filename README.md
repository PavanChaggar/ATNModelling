# ATNModelling

## Installation 

### Julia Installation
This package has been tetsed on Julia 1.11.1 and 1.11.2

If julia is not installed, you can use `juliaup` to manage the installation. 
See the [`juliaup`](https://github.com/JuliaLang/juliaup) page for installation 
instructions. 

Once installed, the relevant `julia` version can be installed using 
```
juliaup add 1.11.2
```

### Package Installation

To install the package and the relevant dependencies, clone with this repo using:

```
git clone https://github.com/PavanChaggar/ATNModelling.git
```

Once downloaded, `cd` into the directory and use the following command to install 
the package and dependencies. 

```
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

Once installed, scripts can be run in the project with the command 
```
julia --project {path/to/script.jl}
```
which will active the project environment and run the script. 

## Data requirements 

We have used data from ADNI and BioFINDER-2 to conduct this study. Data from ADNI is publicly available through [LONI](https://adni.loni.usc.edu).
BioFINDER data is available on request via email to bf_executive@med.lu.se. 

Specifically, we use the AB-PET and tau-PET data analysis derivatives from UC Berekeley. Structural MRI data is taken from matching tau PET scans, provided with the Berekeley tau-PET derivatives. We also use the sMRI analysis from UCSF to obtain total ICV values. 

To fully reproduce the analysis, the corresponding files shoul dbe downloaded and places in a directory names `data/ADNI`. 

The code available here will reproduce all results except for those that depend directly on BioFINDER-2 data. Specifically Figure 2B is only partially reproducible with ADNI data. Code relating to the BioFINDERE-2 analysis can be made available on request. 

## Usage to Reproduce Figures

Figures can be reproduces with the folowing scripts: 
* Figure 1.
    `plots/simulation/atn-simulation.jl`.
    Reproduces Figure 1 and S1. 
* Figure 2.
    `plots/simulation/posterior-predictive-checks.jl`.
    Produces the output for ADNI data only. 
* Figure 3.
    `plots/simulation/simulation/colocalisation.jl`.
* Figure 4.
    `plots/pkpd/spatial-pkpd.jl`.
* Figure 5.
    `plots/pkpd/pkpd.jl`.

## Usage to Reproduce ATN Model 

The ATN model requires four fixed parameters to specify, these are the AB PET healthy baseline values, $A_0$, AB PET carrying capacities $A_\infty$, tau PET baseline values $T_0$ and tau PET PART capacities $\kappa$, all of which are included in this repository. 

The parameters for AB PET can be found in `output/analysis-derivatives/ab-derivatives/fbb/ab-params.csv` for Florbetaben PET and `output/analysis-derivatives/ab-derivatives/fbp/ab-params.csv` for Florbetapir PET. 

The parameters for tau PET can be found in `data/adni-derivatives/tau-params.csv` for baseline tau PET values (calculated previously in Chaggar et al., 2025 PloS Biology), and `output/tau-derivatives/pypart-sym.csv` for the PART carrying capacities.

These are automatically loadable using the `ATNModelling` package as: 
```
using ATNModelling.SimulationUtils: load_ab_params, load_tau_params

v0, vi, part = load_tau_params()
fbb_u0, fbb_ui = load_ab_params(tracer="FBB")
fbp_u0, fbp_ui = load_ab_params(tracer="FBP")
```

The model also requires the graph Laplacian to model tau transport on the connectome. The connectome is loaded via the `Connectomes.jl` package and can be imported within `ATNModelling` as
```
using ATNModelling.ConnectomeUtils: get_connectome
using Connectomes: laplacian_matrix

c = get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2);
L = laplacian_matrix(c) 
```
which returns an $72\times72$ dimensional matrix, corresponding to the 68 cortical DK regions plus the bilateral hippocampus and amygdala. 

The ATN model can then be simulated by
```
using ATNModelling.SimulationUtils: make_atn_model, simulate
atn_model = make_atn_model(fbb_u0, fbb_ui, v0, part, L)

initial_conditions = [fbb_u0 .+ 0.1; v0 .+ 0.1; zeros(72)]
time_span = (0. 30.)

amyloid_production = 0.75
tau_transport = 0.02
tau_production = 0.5 
ab_tau_coupling = 3.21
neurodegen_rate = 0.05

parameters  = [amyloid_production, tau_transport, 
               tau_production, ab_tau_coupling, 
               neurodegen_rate]

atn_sol = simulate(atn_model, 
                    initial_conditions, 
                    time_span, 
                    parameters)
```
which defaults to a `Tsit5()` ODE solver using `DifferentialEquations.jl`. 

The solution can be plotting by adding and loading `Plots.jl` or `CairoMakie.jl` and running
```
using Plots

plot(sol, idxs=1:72) # plots AB 
plot(sol, idxs=73:144) # plots tau 
plot(sol, idxs=145:end) # plots neurodegeneration 
```

## Usage to estimate fixed model parameters
The dATN model has fixed parameters $A_0$, $A_\infty$, $T_0$ and $\kappa$. $T_0$ was estimated previously from a multitau cohort in Chaggar et al., PLoS Biology 2025, and are provided in `data/adni-derivatives/tau-params.csv`. 

The code for deriving $A_0$ and $A_\infty$ can be found in `analysis/data-analysis/ab-analysis.jl`. 

The code for deriving $\kappa$ can be found in `analysis/data-analysis/tau-analysis.jl`. This file contains code to output cross-seciontl ROI data that can be processed with the python script `py-analysis/gaussian-mixture.py` and then compile the results to derive $\kappa$. 

## Reference 
```
@article{chaggar2026dynamical,
  title={Dynamical A $\beta$-Tau-Neurodegeneration Model Predicts Alzheimer’s Disease Mechanisms and Biomarker Progression},
  author={Chaggar, Pavanjit and Vogel, Jacob W and Thompson, Travis B and Aldea, Roxana and Strandberg, Olof and Stomrud, Erik and Palqmvist, Sebastian and Ossenkoppele, Rik and Jbabdi, Saad and Magon, Stefano and others},
  journal={bioRxiv},
  pages={2026--01},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```