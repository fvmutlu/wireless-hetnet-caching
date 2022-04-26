include("helper_functions.jl")

using ForwardDiff

function funcSetup(netgraph::network_graph, reqs::requests, V::Int64, M::Int64, noise::Float64, D_bh_mc::Float64, D_bh_sc::Float64)
    paths = netgraph.paths
    edges = netgraph.edges
    int_edges = netgraph.int_edges
    gains = netgraph.gains
    sc_nodes = netgraph.sc_nodes

    numof_requests = length(reqs.items)
    numof_hops = sum(length.(paths)) - length(paths) # Number of wireless nodes involved in paths, with duplicates, backhaul excluded
    numof_edges = size(edges,1) # Number of wireless edges involved in paths, no duplicates

    # Set up function F
    SINR = Array{Function}(undef, numof_edges) # We need the SINR functions first
    #vprime = edges[:, 1] # Collection of v' nodes to get gains for interference calculation (this includes the actual signal of interest power)
    for edge in 1:numof_edges
        v = edges[edge, 1]; u = edges[edge, 2] # Tx side of edge = v, Rx side = u
        #SINR[edge] = S -> ( gains[v, u] * S[edge] ) / ( noise + sum( gains[vprime, u] .* S[:] ) - gains[v, u] * S[edge] ) # SINR = signal power / (noise + all powers - signal power)
        if isempty(int_edges[edge])
            SINR[edge] = S -> ( gains[v, u] * S[edge] ) / noise
        else
            vprime = [ edges[e, 1] for e in int_edges[edge] ] # Tx side of interferer will be the first node in the edge numbered 'e', where 'e' scans through all items in the interfering edges for the current edge        
            SINR[edge] = S -> ( gains[v, u] * S[edge] ) / ( noise + sum( gains[vprime, u] .* S[int_edges[edge]] ) ) # SINR = signal power / (noise + interfering powers)
        end
    end

    F = Array{Function}(undef, numof_hops)
    f_marker = 0
    for (r, p) in enumerate(paths)
        plen = length(p) - 1 # Exclude backhaul node
        lambda_i_p = reqs.rates[r]
        for k in 1:plen-1
            pk = p[k]
            pk_next = p[k + 1]
            edge = findfirst( [ [pk_next, pk] == edges[i, :] for i in 1:size(edges, 1) ] ) # get edge index for (pₖ₊₁,pₖ), [pk_next, pk] and edges[i, :] are both of type Array{Int64, 1}
            F[f_marker + k] = S -> lambda_i_p * ( 1 / log(1 + SINR[edge](S)) )
        end
        if p[plen] in sc_nodes
            F[f_marker + plen] = S -> lambda_i_p * D_bh_sc
        else
            F[f_marker + plen] = S -> lambda_i_p * D_bh_mc
        end
        f_marker = f_marker + plen;
    end

    # Set up function G
    G = Array{Function}(undef, numof_hops)
    Gprime = Array{Function}(undef, numof_hops)
    Gintegral = Array{Function}(undef, numof_hops)
    gp_marker = 0 # We need to establish G' functions, since the length of paths are variable we need to keep track of where we are in the following loop
    A = zeros(Int64, M*V, M*V) # For affine composition G(Y) = G'(AY)
    for (r, p) in enumerate(paths) # Iterate over all paths and establish G' functions (length(paths) = numof_requests [ = U for current case])
        i = reqs.items[r] # Item i requested by nth request
        plen = length(p) - 1 # Exclude backhaul node
        for k in 1:plen
            pk = p[k]
            Gprime[gp_marker + k] = Y -> 1 - min(Y[ (pk-1)*M + i ], 1) # Establish symbolic function G' for kth node along path p ( 1 - min(y,1) )
            pl = p[1:k]
            Gintegral[gp_marker + k] = Y -> prod(1 .- Y[ (pl .- 1) .* M .+ i])
            A[ (pk-1)*M + i, (pl .- 1) .* M .+ i] .= 1
        end
        gp_marker = gp_marker + plen # Mark all G' functions related to path p as computed
    end    
    G = [ Y -> Gprime[i](A*Y) for i in 1:numof_hops]

    grad_S_F = [ S -> ForwardDiff.gradient(F[i], S) for i in 1:numof_hops ]
    subgrad_Y_G = [ Y ->  ForwardDiff.gradient(G[i], Y) for i in 1:numof_hops] # ForwardDiff takes one extreme of the subdifferential range, just as in the equivalent MATLAB function

    return (F = F, grad_S_F = grad_S_F, G = G, subgrad_Y_G = subgrad_Y_G, Gintegral = Gintegral) # TODO: Make "funcs" tuple into a type-defined struct
end