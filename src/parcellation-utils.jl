module ParcellationUtils 

using Connectomes: Parcellation, node2FS, get_lobe, get_node_id, connectome_path

dktdict = node2FS()

function get_parcellation()
    Parcellation(connectome_path())
end

function get_cortex(parc::Parcellation)
    filter(x -> get_lobe(x) != "subcortex", parc)
end

function get_dkt_names(parc::Parcellation)
    [dktdict[i] for i in get_node_id.(parc)]
end

end