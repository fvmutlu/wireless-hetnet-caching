function symSetup(netgraph, params)
    paths = netgraph.paths
    edges = netgraph.edges
    gains = netgraph.gains
    sc_nodes = netgraph.sc_nodes

    noise = params.noise

    numof_hops = sum(length.(paths)) - 2*length(paths) # Number of wireless edges involved in paths, with duplicates
    numof_edges = size(edges,1) # Number of wireless edges involved in paths, no duplicates
    
    # Form symbolic SINR and f expressions (eq. 1 and 2 in the paper)
    S = [symbols("s_$i") for i in 1:numof_edges]
    SINR = [symbols("sinr_$i") for i in 1:numof_edges]
    for e in 1:numof_edges
        v = edges[e, 1]; u = edges[e, 2] # Tx side of edge = v, Rx side = u
        s_vu = S[e]
        power = gains[v,u]*s_vu
        interference = 0;
        for edge in 1:numof_edges
            vprime = edges[edge, 1] # get Tx side of edge
            s_vprime_u = S[edge]
            interference = interference + gains[vprime, u]*s_vprime_u
        end
        interference = interference - power
        SINR[e] = power / (noise + interference)
    end

    F = [symbols("f_$i") for i in 1:numof_hops]
    f_marker = 0
    for p in paths
        plen = length(p) - 1 # Exclude backhaul node
        lambda_i_p = request_rates(n)
        for k in 1:plen-1
            pk = p[k]
            pk_next = p[k + 1]
            edge = findfirst( [ [pk_next pk] == edges[i,:] for i in 1:size(a,1) ] ) # get edge index for (p_{k+1},p_k)
            F[f_marker + k] = lambda_i_p * ( 1 / log(1 + SINR[edge]) )
        end
        if p[plen] in sc_nodes
            F[f_marker + plen] = lambda_i_p * D_bh_sc
        else
            F[f_marker + plen] = lambda_i_p * D_bh_mc
        end
        f_marker = f_marker + plen;
    end
end