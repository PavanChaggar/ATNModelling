module ConnectomeUtils 

using Connectomes: Parcellation, node2FS, get_lobe, get_node_id, connectome_path

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

end