# ATNModelling

## Installation 

### Julia Installation
This package has been tetsed on Julia 1.11.1 and 1.11.2. 

If julia is not installed, you can use `juliaup` to manage the installation. 
See the [`juliaup`](https://github.com/JuliaLang/juliaup) page for installation 
instructions. 

Once installed, the relevant `julia` version can be installed using 
```
juliaup update
```
which will fetch the latest version of julia (1.11.2). 

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

