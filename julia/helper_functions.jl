using Convex, SCS, Random, Distributions, StatsBase, Dates, Combinatorics, LightGraphs, SimpleWeightedGraphs

struct network_graph # Will add more fields if necessary
    paths::Array{Array{Int64,1},2}
    edges::Array{Int64,2}
    gains::Array{Float64,2}
    sc_nodes::Array{Int64,1}
    V_pos::Array{Float64,2}
    SC_pos::Array{Float64,2}
end

function cellGraph(V::Int64, SC::Int64, R_cell::Float64, pathloss_exponent::Float64, C_mc::Float64, C_sc::Float64)
    U = V-SC-1 # Number of users

    V_pos = zeros(Float64, V, 3) # Last column is node type identifier MC = 0, SC = 1, U = 2
    SC_pos = zeros(Float64, SC, 2) # No need for identifier, all SCs
    R_sc = R_cell / 2 # Initial coverage for SCs, avoid placing another SC in this range

    # Place non-MC nodes
    for v in 2:V
        norm = 0;
        V_pos[v,:] .= 0 # (0,0,0) is the MC entry hence further nodes shouldn't be placed there
        while (norm > R_cell) || ((V_pos[v, 1] == 0) && (V_pos[v, 2] == 0)) # If point is outside cell area or is at the center, find new point
            V_pos[v,1:2] = 2 * R_cell .* rand(Float64,1,2) .- R_cell # Generate point in interval (-R_cell, R_cell) and assign to x for node v's position
            norm = sqrt(V_pos[v, 1]^2 + V_pos[v, 2]^2) # Calculate norm to make sure point is within cell
        end
    end
    V_pos[2:V,3] .= 2 # Mark all non-MC nodes as U first

    # Mark SC nodes
    sc_nodes = zeros(Int64, SC)
    sc_count = 0
    while (sc_count == 0) # Assign random node to be first SC
        rand_node = rand(2:V)
        if sqrt(V_pos[rand_node, 1]^2 + V_pos[rand_node, 2]^2) > R_sc # Mark the node as SC only if it is a certain distance away from the MC (what if all of them are closer?)
            sc_nodes[1] = rand_node; # Assign node ID in sc_nodes array
            SC_pos[1,:] = V_pos[rand_node,[1 2]] # Assign node position in SC positions array
            V_pos[rand_node,3] = 1 # Mark node as SC
            sc_count = 1 # Set sc_count to 1 since we have our first SC
        end
    end
    while sc_count < SC # Assign further nodes to SC set
        for v in 2:V
            if (sc_count < SC) && !(v in sc_nodes) # If there is still need for an SC, get the distance from this node to each SC node
                dist_to_sc = sqrt.(sum((SC_pos .- V_pos[v, [1 2]]) .^ 2, dims=2))
                if all(i->(i > R_sc), dist_to_sc) # If all SC nodes are sufficiently far away, node v can be marked as SC
                    sc_count += 1 # Increment sc_count
                    sc_nodes[sc_count] = v # Assign node ID in sc_nodes array
                    SC_pos[sc_count,:] = V_pos[v,[1 2]] # Assign node position in SC positions array 
                    V_pos[v,3] = 1 # Mark node as SC                
                end
            end    
        end # If for all possible v, no far enough SCs were found, we'll have to adjust expectations
        R_sc = R_sc - R_sc/5 # Decrease R_sc by %20 at each iteration (the while guarantees that if there are enough SC this loop will not repeat)
    end

    # Create the routing graph
    G = SimpleWeightedGraph(V+1)
    non_mc_nodes = 2:V
    u_nodes = zeros(Int64, 1, U)
    u_nodes[1,:] = non_mc_nodes[findall(i->!(i in sc_nodes),non_mc_nodes)] # Create array with U node IDs
    for u in u_nodes # Add an edge from each U to its associated SC
        #dist_to_sc = transpose(sqrt.(sum((SC_pos .- V_pos[u, [1 2]]) .^ 2, dims=2))) # Calculate the distance from this u to all SCs
        dist_to_sc = sqrt.(sum((SC_pos .- V_pos[u, [1 2]]) .^ 2, dims=2)) # Calculate the distance from this u to all SCs
        dist_to_mc = sqrt(sum(V_pos[u, [1 2]] .^ 2, dims=2))[1] # Calculate the distance from this U to MC
        if minimum(dist_to_sc) < dist_to_mc # If the closest SC is closer than MC
            add_edge!(G, u, sc_nodes[argmin(dist_to_sc)], max(1, minimum(dist_to_sc)^pathloss_exponent)) # add_edge!(Graph, Node 1 of Edge, Node 2 of Edge, Cost of Edge)
        else # If MC is closer
            add_edge!(G, u, 1, max(1, dist_to_mc^pathloss_exponent))
        end
    end
    for sc_pair in collect(combinations(sc_nodes,2)) # Add edge between every pair of SCs
        dist = sqrt(sum((V_pos[sc_pair[1], [1 2]] - V_pos[sc_pair[2], [1 2]]) .^ 2, dims=2))[1] # Calculate distance between the two SCs
        add_edge!(G, sc_pair[1], sc_pair[2], max(1, dist^pathloss_exponent)) 
    end
    for sc in sc_nodes # Add edge between MC and each SC, then backhaul (V+1) and each SC
        dist_to_mc = sqrt(sum(V_pos[sc, [1 2]] .^ 2, dims=2))[1] # Calculate the distance between this sc and MC
        add_edge!(G, sc, 1, max(1, dist_to_mc^pathloss_exponent))
        add_edge!(G, sc, V+1, C_sc)
    end
    add_edge!(G, 1, V+1, C_mc) # Add edge between MC and backhaul

    paths = fill(Int64[], 1, U) # Array of arrays for paths (replaces the cell from MATLAB here)
    for u in 1:U
        paths[u] = enumerate_paths(dijkstra_shortest_paths(G,u_nodes[u]),V+1)
    end

    # Determine gains between every pair of nodes for interference calculation purposes
    gains = zeros(Float64, V, V)
    for v in 1:V, u in 1:V
        dist = sqrt(sum((V_pos[v, [1 2]] - V_pos[u, [1 2]]) .^ 2, dims=2))[1] # Calculate distance between the two nodes
        gains[v,u] = min(1, dist ^ (-pathloss_exponent))
    end

    # Extract wireless edges that are involved in paths (Eₚ) (we will call this array 'edges' from now on since this is the set of edges of primary importance)
    numof_edges = sum(length.(paths)) - 2*length(paths) # First we get the total number of edges (excluding backhaul) with potential duplicates so that we can determine the upper bound in order to allocate the array accordingly
    edges = zeros(Int64, numof_edges, 2) # We allocate the edges array using the above number
    marker = 0
    for p in paths
        plen = length(p)-1 # Exclude the backhaul node
        for k = 1:plen-1
            pk = p[k]
            pk_next = p[k+1]
            edges[marker+k, :] = [pk_next, pk]
        end
        marker = marker + plen - 1;
    end
    edges = unique(edges,dims=1)

    netgraph = network_graph(paths, edges, gains, sc_nodes, V_pos, SC_pos)
    return netgraph
end

function incSC(netgraph::network_graph, R_cell::Float64)
    V = size(netgraph.V_pos, 1) + 1
    SC = length(netgraph.sc_nodes) + 1
    R_sc = R_cell / 2
    
    norm = 0;
    new_V_pos = vcat(netgraph.V_pos, [0 0 0])
    new_SC_pos = vcat(netgraph.SC_pos, [0 0])
    while new_V_pos[V,3] == 0 || ((new_V_pos[V, 1] == 0) && (new_V_pos[V, 2] == 0))
        points = 2 * R_cell .* rand(Float64,20,2) .- R_cell # Generate point in interval (-R_cell, R_cell) and assign to x for node v's position
        norm = sqrt.(points[:, 1].^2 + points[:, 2].^2) # Calculate norm to make sure point is within cell
        indices = findall(x -> x < R_cell, norm)
        points = points[indices, :]
        dist_to_sc = [ sqrt.(sum((netgraph.SC_pos .- points[i, [1 2]]) .^ 2, dims=2)) for i in 1:length(indices) ]
        indices = findall(x -> all(y -> y > R_sc, x), dist_to_sc)
        if !isempty(indices)
            points = points[indices, :]
            dist_to_sc = sum.(dist_to_sc[indices])
            ind = argmin(dist_to_sc)
            new_SC_pos[SC, :] = points[ind, [1 2]]
            new_V_pos[V, :] = hcat(points[ind, [1 2]], 1)
        else
            R_sc = R_sc * 0.8
        end              
    end

    return new_SC_pos
end

function randomRequests(pd::Array{Float64,1}, numof_requests::Int64, base_rate::Float64)
    requested_items = zeros(Int64, 1, numof_requests)
    request_rates = zeros(Float64, 1, numof_requests)
    w = ProbabilityWeights(pd) # Convert the probability distribution pd (which defines a discrete random variable that contains all sequential integer values in the range 1:M) where M is the size of the set of items
    for r in 1:numof_requests # Iterate over all requests
        x = sample(1:length(pd),w) # Sample an item from the ordered list of items {1, 2, ..., M-1, M} according to the probability distribution pd
        requested_items[1, r] = x # Assign sampled item as the requested item for this request
        request_rates[1, r] = base_rate + pd[x] # Calculate the request rate (λ) (NOTE: The calculation here is a valid but dumb one, it could be much smarter but that would be determined by the network)
    end
    return requested_items, request_rates
end

function projOpt(S_step_t, Y_step_t, consts)
    P_min = consts.P_min
    P_max = consts.P_max
    C = consts.C
    cache_capacity = consts.cache_capacity

    # Minimum norm subproblem for S projection
    dim_S = size(S_step_t, 1) # length of power vector
    S_proj_t = Variable(dim_S) # problem variable, a column vector (Convex.jl)
    problem = minimize(norm(S_proj_t - S_step_t),[S_proj_t >= P_min, ones(Int64, 1, dim_S)*S_proj_t <= P_max]) # problem definition (Convex.jl)
    solve!(problem, SCS.Optimizer(verbose=false)) # use SCS solver (Convex.jl, SCS.jl)

    # Minimum norm subproblem for Y projection
    dim_Y = size(Y_step_t,1)
    Y_proj_t = Variable(dim_Y)
    problem = minimize(norm(Y_proj_t - Y_step_t),[Y_proj_t >= 0, Y_proj_t <= 1, C*Y_proj_t <= cache_capacity])
    solve!(problem, SCS.Optimizer(verbose=false))
    
    return evaluate(S_proj_t), evaluate(Y_proj_t)
end

function randomInitPoint(dim_S, dim_Y, weight, consts)
    Random.seed!(Dates.value(Dates.now())) # set the seed with current system time
    P_min = consts.P_min
    P_max = consts.P_max

    # Randomized initial point for power vector
    mean = (P_max + P_min)/2 + (P_max + P_min)*weight/2
    var = (P_max + P_min)/10
    pd = Normal(mean, var)
    S_0 = rand(pd, dim_S)

    # for caching vector
    mean = 0.5 + 0.5*weight/2
    var = 0.1
    pd = Normal(mean, var)
    Y_0 = rand(pd, dim_Y)

    return projOpt(S_0, Y_0, consts)
end

function pipageRound(F, Gintegral, S_opt, Y_opt, M, V, cache_capacity)
    Y_matrix = reshape(Y_opt, (M, V)) # put the caching vector into matrix form
    epsilon = 1e-3
    for v in 1:V # repeat for all nodes 
        y = Y_matrix[:,v]; # get node v's caching variables (column vector)
        y[findall(i->(i<epsilon), y)] .= 0 # fix floating-point rounding errors by rounding all values in epsilon neighborhood of 0
        y[findall(i->(abs(i-1)<epsilon), y)] .= 1 # fix floating-point rounding errors by rounding all values in epsilon neighborhood of 1
        while !(all(isbinary, y)) # repeat as long as there are fractional values
            y_frac_pair_ind = findall(!isbinary, y)
            if length(y_frac_pair_ind)==1
                y_temp = y
                y_temp[y_frac_pair_ind] = 1
                if sum(y_temp)<=cache_capacity[v]
                    y[y_frac_pair_ind] = 1
                else
                    y[y_frac_pair_ind] = 0
                end
            else
                DO_best = Inf;
                y_best = [0.0 0.0]

                for pair_ind_temp in collect(combinations(y_frac_pair_ind, 2))
                    y_frac_pair = y[pair_ind_temp]
                    y1 = y_frac_pair[1]
                    y2 = y_frac_pair[2]
                    y_temp = y

                    # Case 1: Try rounding y1 to 0
                    if y1 + y2 < 1
                        y_temp[pair_ind_temp] = [0 y1+y2]
                        Y_temp = Y_matrix
                        Y_temp[:,v] = y_temp
                        Y_temp = vcat(Y_temp...)
                        DO_temp = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ Gintegral[n](Y_temp) for n in 1:length(Gintegral) ])                        
                        if (DO_temp < DO_best)
                            DO_best = DO_temp
                            y_best = y_temp
                        end
                    end

                    # Case 2: Try rounding y1 to 1
                    if y2 - (1 - y1) > 0
                        y_temp[pair_ind_temp] = [1 y2 - (1 - y1)]
                        Y_temp = Y_matrix
                        Y_temp[:,v] = y_temp
                        Y_temp = vcat(Y_temp...)                       
                        DO_temp = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ Gintegral[n](Y_temp) for n in 1:length(Gintegral) ])                        
                        if (DO_temp < DO_best)
                            DO_best = DO_temp
                            y_best = y_temp
                        end
                    end

                    # Case 3: Try rounding y2 to 0
                    if y1 + y2 < 1
                        y_temp[pair_ind_temp] = [y1+y2 0]
                        Y_temp = Y_matrix
                        Y_temp[:,v] = y_temp
                        Y_temp = vcat(Y_temp...)                      
                        DO_temp = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ Gintegral[n](Y_temp) for n in 1:length(Gintegral) ])
                        if (DO_temp < DO_best)
                            DO_best = DO_temp
                            y_best = y_temp
                        end
                    end

                    # Case 4: Try rounding y2 to 1
                    if y1 - (1 - y2) > 0
                        y_temp[pair_ind_temp] = [y1-(1-y2) 1]
                        Y_temp = Y_matrix
                        Y_temp[:,v] = y_temp
                        Y_temp = vcat(Y_temp...)                     
                        DO_temp = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ Gintegral[n](Y_temp) for n in 1:length(Gintegral) ])        
                        if (DO_temp < DO_best)
                            DO_best = DO_temp
                            y_best = y_temp
                        end
                    end
                end
                y = round.(y_best); # If no cases had better than the current objective, this will just pick the best result out of all trials.
                # We round because the results from projection can be arbitrarily close to 1 or 0 while they should be exactly 1 or 0 (e.g. 0.9999999875 or 4.57987363e-11), good old floating point operations
            end
        end
        Y_matrix[:,v] = y
    end
    return vcat(Y_matrix...)
end

function isbinary(x)
    if (x==0 || x==1)
        return true
    else
        return false
    end
end