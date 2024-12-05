module ParcellationUtils 

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

end