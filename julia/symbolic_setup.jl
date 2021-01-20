include("basic_functions.jl")
include("helper_functions.jl")

function symSetup(netgraph, params)
    paths = netgraph.paths
    edges = netgraph.edges
    gains = netgraph.gains
    sc_nodes = netgraph.sc_nodes

    noise = params.noise
    numof_requests = params.numof_requests
    V = params.V
    M = params.M
    D_bh_mc = params.D_bh_mc
    D_bh_sc = params.D_bh_sc

    numof_hops = sum(length.(paths)) - length(paths) # Number of wireless nodes involved in paths, with duplicates, backhaul excluded
    numof_edges = size(edges,1) # Number of wireless edges involved in paths, no duplicates

    pd = fill(1/M, 5) # uniform dist
    requested_items, request_rates = randomRequests(pd, numof_requests, 1.0)

    SINR = Array{Function}(undef, numof_edges)
    vprime = edges[:, 1] # Collection of v' nodes to get gains for interference calculation (this includes the actual signal of interest power)
    for e in 1:numof_edges
        v = edges[e, 1]; u = edges[e, 2] # Tx side of edge = v, Rx side = u
        SINR[e] = S -> ( gains[v, u] * S[e] ) / ( noise + sum( gains[vprime, u] .* S[:] ) - gains[v, u] * S[e] ) # SINR = signal power / (noise + all powers - signal power)
    end

    F = Array{Function}(undef, numof_hops)
    #=
    # Form symbolic SINR and f expressions (eq. 1 and 2 in the paper)
    S = [ symbols("s_$i") for i in 1:numof_edges ]
    SINR = [ symbols("sinr_$i") for i in 1:numof_edges ]    
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

    F = [ symbols("f_$i") for i in 1:numof_hops ]
    f_marker = 0
    r = 1
    for p in paths
        plen = length(p) - 1 # Exclude backhaul node
        lambda_i_p = request_rates[1, r]
        for k in 1:plen-1
            pk = p[k]
            pk_next = p[k + 1]
            edge = findfirst( [ [pk_next, pk] == edges[i, :] for i in 1:size(edges, 1) ] ) # get edge index for (pₖ₊₁,pₖ), [pk_next, pk] and edges[i, :] are both of type Array{Int64, 1}
            F[f_marker + k] = lambda_i_p * ( 1 / log(1 + SINR[edge]) )
        end
        if p[plen] in sc_nodes
            F[f_marker + plen] = lambda_i_p * D_bh_sc
        else
            F[f_marker + plen] = lambda_i_p * D_bh_mc
        end
        f_marker = f_marker + plen;
        r += 1
    end

    ## Form symbolic caching expressions
    Y = [ symbols("y_$i") for i in 1:(M*V) ] # Here Y is actually vec(Y)^T from the paper
    Gprime = [ symbols("gp_$i") for i in 1:numof_hops ]
    Gintegral = [ symbols("gint_$i") for i in 1:numof_hops ]
    gp_marker = 0 # We need to establish G' functions, since the length of paths are variable we need to keep track of where we are in the following loop
    A = zeros(Int64, M*V, M*V) # For affine composition G(Y) = G'(AY)
    for r in 1:numof_requests # Iterate over all paths (using iterator n) and establish G' functions (length(paths) = length(requests) = U)
        i = requested_items[r] # Item i requested by nth request
        p = paths[r] # Path p for nth request
        plen = length(p) - 1 # Exclude backhaul node
        for k in 1:plen
            pk = p[k]
            y_pk_i = Y[ (pk-1)*M + i ]
            Gprime[gp_marker + k] = 0.5 * (-y_pk_i * sign(1-y_pk_i) - sign(y_pk_i-1) - y_pk_i + 1) # Establish symbolic function G' for kth node along path p ( 1 - min(y,1) )
            Gintegral[gp_marker + k] = 1
            for l in 1:k
                pl = p[l]
                A[ (pk-1)*M + i, (pl-1)*M + i] = 1 # In matrix A, mark entries corresponding to all nodes on path p before pk as 1
                y_pl_i = Y[ (pl-1)*M + i] # ROUNDING TEST
                Gintegral[gp_marker + k] = Gintegral[gp_marker + k] * (1-y_pl_i) # ROUNDING TEST
            end
        end
        gp_marker = gp_marker + plen # Mark all G' functions related to path p as computed
    end
    G = matSubs(Gprime, Y, A*Y)

    C = zeros(Int64, V, M*V) # For cache capacity constraint C*Y <= cache_capacity
    for n in 1:V
        C[n, (n-1)*M + 1 : n*M] .= 1 # In matrix C, mark entries corresponding to node n's items as 1
    end =#

    return SINR
end