module ConnectomeUtils 

using Connectomes: Connectome, Parcellation, 
                   connectome_path, node2FS, get_node_id, 
                   get_lobe, filter, laplacian_matrix, slice,
                   FS2Connectome, get_hemisphere, get_coords
using DrWatson: datadir
using FileIO
using LinearAlgebra

dktdict = node2FS()
"""
    get_parcellation()

Return a `Connectomes.Parcellation` for the DKT atlas.
"""
function get_parcellation()
    Parcellation(connectome_path())
end

"""
   get_cortex(parc::Parcellation)

Filter the `parc` for only cortical regions.
"""
function get_cortex(parc::Parcellation)
    filter(x -> get_lobe(x) != "subcortex", parc)
end

"""
   get_dkt_names(parc::Parcellation)

Generate regional names according to FreeSurfer output
"""
function get_dkt_names(parc::Parcellation)
    [dktdict[i] for i in get_node_id.(parc)]
end

"""
    get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2)

Load a `Connectome` with parameters specified by keyword-arguments. 
"""
function get_connectome(;include_subcortex=false, apply_filter=true, filter_cutoff=1e-2)
    c = Connectome(connectome_path())
    if include_subcortex
        if apply_filter
            fc = filter(c, filter_cutoff)
            return fc
        else
            return c
        end
    else
        cortex_parc = filter(x -> get_lobe(x) != "subcortex", c.parc)
        sc = slice(c, cortex_parc)
        if apply_filter
            fc = filter(sc, filter_cutoff)
            return fc
        else
            return sc
        end
    end 
end

_fsdict = FS2Connectome()
_getbraak(fs_regions) = [_fsdict[i] for i in fs_regions]

"""
    get_braak_regions()

Returns a Vector{Vector{Int64}} with each Vector{Int64} corresponding to a set of Braak regions. 
The first Vector{Vector{Int64}} corresponds to the first Braak region and so on. 
"""
function get_braak_regions()
    braak_dict = load(datadir("derivatives/braak-dict.jld2"))
    ks = ["1", "2/3", "4", "5", "6"]
    fs_braak_stages = [braak_dict[k] for k in ks]    
    return map(_getbraak, fs_braak_stages)
end


"""
    distance(x::Vector{Float64}, y::Vector{Float64})

Calculate the Euclidean distance between two points.
"""
distance(x::Vector{Float64}, y::Vector{Float64}) = norm(x .- y, 2)

"""
    get_distance_laplacian(; hemisphere = "right")

Returns a Matrix{Float64} corresponding to the Laplacian matrix of a graph G = (V, E), where v in V 
are regions in the DK atlas and e in E are edges corresponding to inverse squared Euclidean distance.
"""
function get_distance_laplacian(; hemisphere = "right")
    parc = get_parcellation() |> get_cortex
    cortex = filter(x -> get_hemisphere(x) == hemisphere, parc)

    coords = get_coords(cortex)

    distance_matrix = reduce(hcat, [[distance(coords[j,:], coords[i,:]) for i in 1:36] for j in 1:36])
    inv_distance_matrix = replace(1 ./ (distance_matrix).^2, Inf => 0)
    _D = inv_distance_matrix ./ maximum(inv_distance_matrix)
    Ld = diagm(_D * ones(36)) - _D

    return Ld
end

end