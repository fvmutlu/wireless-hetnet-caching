using Convex, ECOS, Random, Distributions, StatsBase, Dates, Combinatorics, DataStructures, Graphs, SimpleWeightedGraphs, GeometryBasics, VoronoiCells, Plots

struct network_graph # Will add more fields if necessary
    paths::Array{Array{Int64,1},2} # Why does this have 2 on the second dimension instead of 1?
    edges::Array{Int64,2}
    int_edges::Array{Array{Int64,1},1}
    gains::Array{Float64,2}
    sc_nodes::Array{Int64,1}
end

struct requests
    items::Array{Int64,1}
    rates::Array{Float64,1}
end

#= struct constraints
    P_min::Float64
    P_max::Float64
    cache_capacity::Array{Int64,1}
    C::Array{Int64,2}
end =#

struct constraints
    P_min::Float64
    P_max::Float64
    cache_capacity::Array{Int64,1}
    C::Array{Int64,2}
    P::Array{Int64,2}
end

function cellDist(V::Int64, SC::Int64, R_cell::Float64)
    U = V-SC-1 # Number of users

    V_pos = zeros(Float64, V, 3) # Last column is node type identifier MC = 0, SC = 1, U = 2
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
    sc_count = 0
    while (sc_count == 0) # Assign random node to be first SC
        rand_node = rand(2:V)
        if sqrt(V_pos[rand_node, 1]^2 + V_pos[rand_node, 2]^2) > R_sc # Mark the node as SC only if it is a certain distance away from the MC (what if all of them are closer?)
            # sc_nodes[1] = rand_node; # Assign node ID in sc_nodes array
            # SC_pos[1,:] = V_pos[rand_node,[1 2]] # Assign node position in SC positions array
            V_pos[rand_node,3] = 1 # Mark node as SC
            sc_count = 1 # Set sc_count to 1 since we have our first SC
        end
    end

    while sc_count < SC # Assign further nodes to SC set
        sc_nodes = findall(i -> i == 1, V_pos[:,3])
        SC_pos = V_pos[sc_nodes, 1:2] # SC_pos contains points from V_pos that were marked as SCs (3rd column is 1)
        for v in 2:V
            if (sc_count < SC) && !(v in sc_nodes) # If there is still need for an SC, get the distance from this node to each SC node
                dist_to_sc = sqrt.(sum((SC_pos .- V_pos[v, [1 2]]) .^ 2, dims=2))
                if all(i->(i > R_sc), dist_to_sc) # If all SC nodes are sufficiently far away, node v can be marked as SC
                    sc_count += 1 # Increment sc_count
                    # sc_nodes[sc_count] = v # Assign node ID in sc_nodes array
                    # SC_pos[sc_count,:] = V_pos[v,[1 2]] # Assign node position in SC positions array 
                    V_pos[v,3] = 1 # Mark node as SC                
                end
            end    
        end # If for all possible v, no far enough SCs were found, we'll have to adjust expectations
        R_sc = R_sc - R_sc/5 # Decrease R_sc by %20 at each iteration (the while guarantees that if there are enough SC this loop will not repeat)
    end

    return V_pos
end

function voronoiDist(V::Int64, SC::Int64, R_cell::Float64)
    V_pos = zeros(Float64, V, 3)
    sc_nodes = collect(2:SC+1)
    u_nodes = collect(SC+2:V)

    N = 50
    
    rect = Rectangle(Point2(0, 0), Point2(1, 1))
    points = [Point2(rand(), rand()) for p in 1:SC]
    centroids = deepcopy(points)

    for iter in 1:N
        tess = voronoicells(centroids, rect)
        areas = voronoiarea(tess)
        for p in 1:length(points)
            Cx = 0
            Cy = 0
            for c in 1:length(tess.Cells[p])
                vx = c
                vx1 = c % length(tess.Cells[p]) + 1
                Cx += (tess.Cells[p][vx][1] + tess.Cells[p][vx1][1]) * (tess.Cells[p][vx][1]*tess.Cells[p][vx1][2] - tess.Cells[p][vx1][1]*tess.Cells[p][vx][2])
                Cy += (tess.Cells[p][vx][2] + tess.Cells[p][vx1][2]) * (tess.Cells[p][vx][1]*tess.Cells[p][vx1][2] - tess.Cells[p][vx1][1]*tess.Cells[p][vx][2])
            end
            Cx = Cx / (6 * areas[p])
            Cy = Cy / (6 * areas[p])
            centroids[p] = Point2(Cx,Cy)
        end
        #= p = scatter()
        p = scatter(centroids, markersize = 6, label = "Centroids")
        annotate!([(centroids[n][1] + 0.02, centroids[n][2] + 0.03, Plots.text(n)) for n in 1:length(centroids)])
        plot!(tess, legend = :topleft)
        display(p)
        sleep(0.1) =#
    end

    for (p, sc_node) in enumerate(sc_nodes)
        pt = collect(centroids[p]) .- 0.5
        pt = pt .* (R_cell/0.5)
        #points[p] = Point2(pt)        
        V_pos[sc_node,1:2] = pt
        V_pos[sc_node,3] = 1
    end

    upoints = [Point2(rand(), rand()) for p in 1:(V-SC-1)]
    range = 0.5*R_cell/sqrt(SC)
    for (u,u_node) in enumerate(u_nodes)
        c = (u_node-1) % SC + 1
        sc_node = sc_nodes[c]
        V_pos[u_node,1:2] = V_pos[sc_node,1:2] + rand(-range:0.01:range,2)
        #upoints[u] = Point2(V_pos[u_node,1:2])
        V_pos[u_node,3] = 2        
    end

    #= sleep(2.0)
    p = scatter()
    p = scatter(points, markersize = 6, label = "SCs")
    p = scatter(upoints, markersize = 3, label = "Users")
    display(p) =#

    return V_pos    
end

function voronoiMod(V_pos::Array{Float64,2}, SC::Int64, R_cell::Float64)
    sc_nodes = findall(i -> i == 1, V_pos[:,3])       

    N = 50
    
    rect = Rectangle(Point2(0, 0), Point2(1, 1))
    points = [Point2(rand(), rand()) for p in 1:SC]
    centroids = deepcopy(points)

    for iter in 1:N
        tess = voronoicells(centroids, rect)
        areas = voronoiarea(tess)
        for p in 1:length(points)
            Cx = 0
            Cy = 0
            for c in 1:length(tess.Cells[p])
                vx = c
                vx1 = c % length(tess.Cells[p]) + 1
                Cx += (tess.Cells[p][vx][1] + tess.Cells[p][vx1][1]) * (tess.Cells[p][vx][1]*tess.Cells[p][vx1][2] - tess.Cells[p][vx1][1]*tess.Cells[p][vx][2])
                Cy += (tess.Cells[p][vx][2] + tess.Cells[p][vx1][2]) * (tess.Cells[p][vx][1]*tess.Cells[p][vx1][2] - tess.Cells[p][vx1][1]*tess.Cells[p][vx][2])
            end
            Cx = Cx / (6 * areas[p])
            Cy = Cy / (6 * areas[p])
            centroids[p] = Point2(Cx,Cy)
        end
    end

    for (p, sc_node) in enumerate(sc_nodes)
        pt = collect(centroids[p]) .- 0.5
        pt = pt .* (R_cell/0.5)        
        V_pos[sc_node,1:2] = pt
        V_pos[sc_node,3] = 1
    end

    return V_pos    
end

function addVoronoiSC(V_pos::Array{Float64,2}, R_cell::Float64)
    sc_nodes = findall(i -> i == 1, V_pos[:,3])
    V = size(V_pos, 1) + 1
    sc_nodes = vcat(sc_nodes, V)
    V_pos = vcat(V_pos, hcat(0,0,1))
    return voronoiMod(V_pos,length(sc_nodes),R_cell)
end

function addSC(V_pos::Array{Float64,2}, R_cell::Float64) # TODO: Whenever this is called from the simulations, V in parameters must be incremented
    V = size(V_pos, 1) + 1
    sc_nodes = findall(i -> i == 1, V_pos[:,3])
    SC_pos = V_pos[sc_nodes, 1:2]
    R_sc = R_cell / 2
    
    norm = 0
    cond = true
    while cond
        points = 2 * R_cell .* rand(Float64,20,2) .- R_cell # Generate 20 random points in interval (-R_cell, R_cell)
        norm = sqrt.(points[:, 1].^2 + points[:, 2].^2) # Calculate norms
        indices = findall(x -> x < R_cell, norm) # Extract indices of points that fall within the circle
        points = points[indices, :]
        dist_to_sc = [ sqrt.(sum((SC_pos .- points[i, [1 2]]) .^ 2, dims=2)) for i in 1:length(indices) ] # Calculate distance from each point to all existing SCs
        indices = findall(x -> all(y -> y > R_sc, x), dist_to_sc) # Extract indices of points that are enough of a distance away from all existing SCs
        if !isempty(indices)
            points = points[indices, :]
            dist_to_sc = sum.(dist_to_sc[indices])
            ind = argmin(dist_to_sc) # Pick the point with the least sum of distances to all existing SCs (from among those that satisfy the "far enough away" condition as above)
            V_pos = vcat(V_pos, hcat(points[ind, [1 2]], 1))
            cond = false
        else
            R_sc = R_sc * 0.8
        end              
    end
    
    return V_pos
end

function makeGraph(V_pos::Array{Float64,2}, pathloss_exponent::Float64, interference_range::Float64, C_bh_mc::Float64, C_bh_sc::Float64)
    # Create the routing graph
    V = size(V_pos,1)
    sc_nodes = findall(i -> i == 1, V_pos[:,3])
    SC = length(sc_nodes)
    SC_pos = V_pos[sc_nodes, 1:2]
    U = V - SC - 1

    G = SimpleWeightedGraph(V+1)

    non_mc_nodes = collect(2:V)
    u_nodes = zeros(Int64, 1, U)
    u_nodes[1,:] = non_mc_nodes[findall(i->!(i in sc_nodes),non_mc_nodes)] # Create array with U node IDs
    for u in u_nodes # Add an edge from each U to its associated SC
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
        add_edge!(G, sc, V+1, C_bh_sc)
    end
    add_edge!(G, 1, V+1, C_bh_mc) # Add edge between MC and backhaul

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
    numof_edges = size(edges, 1)

    #int_edges = Array{Array{Int64,1},1}(undef, numof_edges)
    int_edges = [ [] for _ in 1:numof_edges ]
    for i in 1:numof_edges, j in 1:numof_edges
        if i != j
            u = edges[i, 2]
            v = edges[j, 1]
            dist = sqrt(sum((V_pos[v, [1 2]] - V_pos[u, [1 2]]) .^ 2, dims=2))[1]
            if dist <= interference_range
                append!(int_edges[i],j)
            end
        end                  
    end

    println(int_edges)

    netgraph = network_graph(paths, edges, int_edges, gains, sc_nodes)
    return netgraph
end

function cellGraph(V::Int64, SC::Int64, R_cell::Float64, pathloss_exponent::Float64, C_bh_mc::Float64, C_bh_sc::Float64)
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
        add_edge!(G, sc, V+1, C_bh_sc)
    end
    add_edge!(G, 1, V+1, C_bh_mc) # Add edge between MC and backhaul

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

    netgraph = network_graph(paths, edges, gains, sc_nodes)
    return netgraph
end

function randomRequests(pd::Array{Float64,1}, numof_requests::Int64, base_rate::Float64)
    requested_items = zeros(Int64, numof_requests)
    request_rates = zeros(Float64, numof_requests)
    w = ProbabilityWeights(pd) # Convert the probability distribution pd (which defines a discrete random variable that contains all sequential integer values in the range 1:M) where M is the size of the set of items
    for r in 1:numof_requests # Iterate over all requests
        x = sample(1:length(pd),w) # Sample an item from the ordered list of items {1, 2, ..., M-1, M} according to the probability distribution pd
        requested_items[r] = x # Assign sampled item as the requested item for this request
        request_rates[r] = base_rate + pd[x] # Calculate the request rate (λ) (NOTE: The calculation here is a valid but dumb one, it could be much smarter but that would be determined by the network)
    end
    return requests(requested_items, request_rates)
end

function dependentRequests(time_slot::Int64, period::Int64, time_constants::Array{Int64,2}, pd::Array{Float64,1}, numof_requests::Int64, base_rate::Float64)
    requested_items = zeros(Int64, numof_requests)
    request_rates = zeros(Float64, numof_requests)

    period_slot = (time_slot - 1) % period

    for r in 1:numof_requests
        w = (1 ./ time_constants[:,r]) .* exp.(-period_slot ./ time_constants[:,r])
        w = w .* pd
        w = ProbabilityWeights(w ./ sum(w))
        x = sample(1:length(pd),w) # Sample an item from the ordered list of items {1, 2, ..., M-1, M} according to the probability distribution pd
        requested_items[r] = x # Assign sampled item as the requested item for this request
        request_rates[r] = base_rate + pd[x] # Calculate the request rate (λ) (NOTE: The calculation here is a valid but dumb one, it could be much smarter but that would be determined by the network)
    end

    if time_slot % period == 0
        new_tc = rand(Int64(floor(period/4)):Int64(ceil(3*period/4)), length(pd),numof_requests)
    else
        new_tc = time_constants;
    end

    return requests(requested_items, request_rates), new_tc
end

function averageRequests(T::Int64, period::Int64, pd::Array{Float64,1}, numof_requests::Int64, base_rate::Float64)
    items_large = Array{Int64,2}(undef, T,numof_requests)
    rates_large = Array{Float64,2}(undef, T,numof_requests)
    tc = rand(Int64(floor(period/4)):Int64(ceil(3*period/4)), length(pd),numof_requests)
    for t in 1:T
        reqs,tc = dependentRequests(t, period, tc, pd, numof_requests, base_rate)
        items_large[t,:] = reqs.items
        rates_large[t,:] = reqs.rates
    end
    requested_items = [ StatsBase.mode(items_large[:,r]) for r in 1:numof_requests ]
    request_rates = [ mean(rates_large[:,r]) for r in 1:numof_requests ]
    return requests(requested_items, request_rates), items_large, rates_large
end

#= function makeConsts(V::Int64, M::Int64, c_mc::Int64, c_sc::Int64, sc_nodes::Array{Int64,1}, P_min::Float64, P_max::Float64) # TODO: Add per-node or other types of constraints?
    cache_capacity = zeros(Int64,V) # For cache capacity constraint C*Y <= cache_capacity
    cache_capacity[1] = c_mc
    cache_capacity[sc_nodes] .= c_sc

    C = zeros(Int64, V,M*V) # For cache capacity constraint C*Y <= cache_capacity
    for n in 1:V
        C[ n, (n-1)*M+1 : n*M ] .= 1 # In matrix C, mark entries corresponding to node n's items as 1
    end

    return constraints(P_min, P_max, cache_capacity, C)
end =#

function makeConsts(V::Int64, M::Int64, sc_nodes::Array{Int64,1}, edges::Array{Int64,2}, c_mc::Int64, c_sc::Int64, P_min::Float64, P_max::Float64) # TODO: Add per-node or other types of constraints?
    cache_capacity = zeros(Int64,V) # For cache capacity constraint C*Y <= cache_capacity
    cache_capacity[1] = c_mc
    cache_capacity[sc_nodes] .= c_sc

    C = zeros(Int64, V,M*V) # For cache capacity constraint C*Y <= cache_capacity
    for n in 1:V
        C[ n, (n-1)*M+1 : n*M ] .= 1 # In matrix C, mark entries corresponding to node n's items as 1
    end

    P = zeros(Int64, V,size(edges,1))
    for e in 1:size(edges,1)
        tx_node = edges[e,1]
        P[tx_node, e] = 1
    end

    return constraints(P_min, P_max, cache_capacity, C, P)
end

function projOpt(S_step_t, Y_step_t, consts::constraints)
    P_min = consts.P_min
    P_max = consts.P_max
    C = consts.C
    P = consts.P
    cache_capacity = consts.cache_capacity

    # Minimum norm subproblem for S projection
    if S_step_t == 0 # Skip power optimization if all zeros were passed (relevant for ALT)
        S_proj_t = 0
    else
        dim_S = size(S_step_t, 1) # length of power vector
        S_proj_t = Variable(dim_S) # problem variable, a column vector (Convex.jl)
        #problem = minimize(norm(S_proj_t - S_step_t),[S_proj_t >= P_min, ones(Int64, 1, dim_S)*S_proj_t <= P_max]) # problem definition (Convex.jl), total power constraint
        problem = minimize(norm(S_proj_t - S_step_t),[S_proj_t >= P_min, P*S_proj_t <= P_max]) # problem definition (Convex.jl), total power constraint
        #problem = minimize(norm(S_proj_t - S_step_t),[S_proj_t >= P_min, S_proj_t <= P_max]) # problem definition (Convex.jl), per-transmission power constraint
        #solve!(problem, SCS.Optimizer(), verbose=false) # use SCS solver (Convex.jl, SCS.jl)
        solve!(problem, ECOS.Optimizer, verbose=false; silent_solver=true) # use ECOS solver (Convex.jl, ECOS.jl)
        S_proj_t = evaluate(S_proj_t)
    end

    # Minimum norm subproblem for Y projection
    if Y_step_t == 0
        Y_proj_t = 0
    else
        dim_Y = size(Y_step_t,1)
        Y_proj_t = Variable(dim_Y)
        problem = minimize(norm(Y_proj_t - Y_step_t),[Y_proj_t >= 0, Y_proj_t <= 1, C*Y_proj_t <= cache_capacity])
        #solve!(problem, SCS.Optimizer(), verbose=false)
        solve!(problem, ECOS.Optimizer, verbose=false; silent_solver=true) # use ECOS solver (Convex.jl, ECOS.jl)
        Y_proj_t = evaluate(Y_proj_t)
    end
    return S_proj_t, Y_proj_t
end

#= function projOpt(S_step_t, Y_step_t, consts::constraints)
    P_min = consts.P_min
    P_max = consts.P_max
    C = consts.C
    cache_capacity = consts.cache_capacity
    

    # Minimum norm subproblem for S projection
    if S_step_t == 0 # Skip power optimization if all zeros were passed (relevant for ALT)
        S_proj_t = 0
    else
        dim_S = size(S_step_t, 1) # length of power vector
        S_proj_t = Variable(dim_S) # problem variable, a column vector (Convex.jl)
        problem = minimize(norm(S_proj_t - S_step_t),[S_proj_t >= P_min, ones(Int64, 1, dim_S)*S_proj_t <= P_max]) # problem definition (Convex.jl), total power constraint
        #problem = minimize(norm(S_proj_t - S_step_t),[S_proj_t >= P_min, S_proj_t <= P_max]) # problem definition (Convex.jl), per-transmission power constraint
        solve!(problem, SCS.Optimizer(verbose=false), verbose=false) # use SCS solver (Convex.jl, SCS.jl)
        S_proj_t = evaluate(S_proj_t)
    end

    # Minimum norm subproblem for Y projection
    if Y_step_t == 0
        Y_proj_t = 0
    else
        idx = findall(i -> cache_capacity[i] > 0, 1:length(cache_capacity))
        Y_temp = deepcopy(reshape(Y_step_t, (convert(Int64,size(Y_step_t,1)/length(cache_capacity)), length(cache_capacity))))
        Y_temp = Y_temp[:,idx]
        Y_temp = vcat(Y_temp...)
        dim_Y = length(Y_temp)
        Y_proj_t = Variable(dim_Y)
        problem = minimize(norm(Y_proj_t - Y_temp),[Y_proj_t >= 0, Y_proj_t <= 1, ones(Int,length(idx),dim_Y)*Y_proj_t <= cache_capacity[idx]])
        solve!(problem, SCS.Optimizer(verbose=false), verbose=false)
        Y_proj_t = evaluate(Y_proj_t)
        Y_temp = zeros(Float64, convert(Int64,size(Y_step_t,1)/length(cache_capacity)), length(cache_capacity))
        Y_temp[:,idx] = reshape(Y_proj_t, convert(Int64,size(Y_step_t,1)/length(cache_capacity)), length(idx))
        Y_proj_t = vcat(Y_temp...)
    end
    return S_proj_t, Y_proj_t
end =# # ATTEMPT TO SPEED UP CACHE OPTIMIZATION BY REMOVING SPARSE PART OF Y

function randomInitPoint(dim_S, dim_Y, weight, consts)
    Random.seed!(Dates.value(Dates.now())) # set the seed with current system time
    P_min = consts.P_min
    P_max = consts.P_max

    # Randomized initial point for power vector
    mean = (P_max + P_min)/2 + (P_max + P_min)*weight/2
    var = (P_max + P_min)/10
    pd = Distributions.Normal(mean, var)
    S_0 = rand(pd, dim_S)

    # for caching vector
    mean = 0.5 + 0.5*weight/2
    var = 0.1
    pd = Distributions.Normal(mean, var)
    Y_0 = rand(pd, dim_Y)

    return projOpt(S_0, Y_0, consts)
end

#= function pipageRounding(F, G, S_opt, Y_opt, M, V, cache_capacity)
    Y_matrix = deepcopy(reshape(Y_opt, (M, V)))
    Y_matrix[findall(i->(i<0.01), Y_matrix)] .= 0 
    Y_matrix[findall(i->(i>0.99), Y_matrix)] .= 1

    for v in findall(i -> cache_capacity[i] > 0, 1:V)
        while length(findall(!isbinary, Y_matrix[:,v])) > 1
            for pair_ind_temp in collect(combinations(findall(!isbinary, Y_matrix[:,v]), 2))
                y1 = Y_matrix[pair_ind_temp[1],v]
                y2 = Y_matrix[pair_ind_temp[2],v]

                if y1 + y2 < 1
                    Y_matrix[pair_ind_temp[1],v] = 0
                    Y_matrix[pair_ind_temp[2],v] = y1+y2
                    obj1 = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ G[n](vcat(Y_matrix...)) for n in 1:length(G) ])
                    Y_matrix[pair_ind_temp[1],v] = y1+y2
                    Y_matrix[pair_ind_temp[2],v] = 0
                    obj2 = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ G[n](vcat(Y_matrix...)) for n in 1:length(G) ])
                    if obj1 < obj2
                        Y_matrix[pair_ind_temp[1],v] = 0
                        Y_matrix[pair_ind_temp[2],v] = y1+y2
                    end
                else
                    Y_matrix[pair_ind_temp[1],v] = 1
                    Y_matrix[pair_ind_temp[2],v] = y1+y2 - 1
                    obj1 = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ G[n](vcat(Y_matrix...)) for n in 1:length(G) ])
                    Y_matrix[pair_ind_temp[1],v] = y1+y2 - 1
                    Y_matrix[pair_ind_temp[2],v] = 1
                    obj2 = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ G[n](vcat(Y_matrix...)) for n in 1:length(G) ])
                    if obj1 < obj2
                        Y_matrix[pair_ind_temp[1],v] = 1
                        Y_matrix[pair_ind_temp[2],v] = y1+y2 - 1
                    end
                end
            end
        end
        if length(findall(!isbinary, Y_matrix[:,v])) == 1
            id = findfirst(!isbinary, Y_matrix[:,v])
            if round(sum(Y_matrix[:,v]) - Y_matrix[id,v]) + 1 <= cache_capacity[v]
                Y_matrix[ findfirst(!isbinary, Y_matrix[:,v]), v ] = 1
            else
                Y_matrix[ findfirst(!isbinary, Y_matrix[:,v]), v ] = 0
            end
        end
    end

    return vcat(Y_matrix...)
end =#

function pipageRounding(F, G, S_opt, Y_opt, M, V, cache_capacity)
    Y_matrix = deepcopy(reshape(Y_opt, (M, V)))
    Y_matrix[findall(i->(i<0.01), Y_matrix)] .= 0 
    Y_matrix[findall(i->(i>0.99), Y_matrix)] .= 1
    opt_obj = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ G[n](vcat(Y_matrix...)) for n in 1:length(G) ])
    tolerance = 0.025
    for v in findall(i -> cache_capacity[i] > 0, 1:V)
        while length(findall(!isbinary, Y_matrix[:,v])) > 1
            pair_ind_temp = StatsBase.sample(findall(!isbinary, Y_matrix[:,v]), 2, replace=false)
            y1 = Y_matrix[pair_ind_temp[1],v]
            y2 = Y_matrix[pair_ind_temp[2],v]

            if y1 + y2 < 1
                Y_matrix[pair_ind_temp[1],v] = 0
                Y_matrix[pair_ind_temp[2],v] = y1+y2
                obj1 = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ G[n](vcat(Y_matrix...)) for n in 1:length(G) ])
                Y_matrix[pair_ind_temp[1],v] = y1+y2
                Y_matrix[pair_ind_temp[2],v] = 0
                obj2 = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ G[n](vcat(Y_matrix...)) for n in 1:length(G) ])
                #println((opt_obj,obj1,obj2))                
                if (obj1 < obj2) && (obj1 - opt_obj < opt_obj * tolerance)
                    opt_obj = obj1
                    Y_matrix[pair_ind_temp[1],v] = 0
                    Y_matrix[pair_ind_temp[2],v] = y1+y2
                elseif obj2 - opt_obj < opt_obj * tolerance
                    opt_obj = obj2
                    Y_matrix[pair_ind_temp[1],v] = y1+y2
                    Y_matrix[pair_ind_temp[2],v] = 0
                else
                    Y_matrix[pair_ind_temp[1],v] = y1
                    Y_matrix[pair_ind_temp[2],v] = y2
                end
            else
                Y_matrix[pair_ind_temp[1],v] = 1
                Y_matrix[pair_ind_temp[2],v] = y1+y2 - 1
                obj1 = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ G[n](vcat(Y_matrix...)) for n in 1:length(G) ])
                Y_matrix[pair_ind_temp[1],v] = y1+y2 - 1
                Y_matrix[pair_ind_temp[2],v] = 1
                obj2 = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ G[n](vcat(Y_matrix...)) for n in 1:length(G) ])
                #println((opt_obj,obj1,obj2))
                if (obj1 < obj2) && (obj1 - opt_obj < opt_obj * tolerance)
                    opt_obj = obj1
                    Y_matrix[pair_ind_temp[1],v] = 1
                    Y_matrix[pair_ind_temp[2],v] = y1+y2 - 1
                elseif obj2 - opt_obj < opt_obj * tolerance
                    opt_obj = obj2
                    Y_matrix[pair_ind_temp[1],v] = y1+y2 - 1
                    Y_matrix[pair_ind_temp[2],v] = 1
                else
                    Y_matrix[pair_ind_temp[1],v] = y1
                    Y_matrix[pair_ind_temp[2],v] = y2
                end
            end
        end
        if length(findall(!isbinary, Y_matrix[:,v])) == 1
            id = findfirst(!isbinary, Y_matrix[:,v])
            if round(sum(Y_matrix[:,v]) - Y_matrix[id,v]) + 1 <= cache_capacity[v]
                Y_matrix[ findfirst(!isbinary, Y_matrix[:,v]), v ] = 1
            else
                Y_matrix[ findfirst(!isbinary, Y_matrix[:,v]), v ] = 0
            end
        end
    end

    return vcat(Y_matrix...)
end

function pipageRound(F, Gintegral, S_opt, Y_opt, M, V, cache_capacity) # TODO: consider replacing this with another rounding function
    Y_matrix = deepcopy(reshape(Y_opt, (M, V))) # put the caching vector into matrix form
    epsilon = 1e-3
    for v in 1:V # repeat for all nodes 
        y = deepcopy(Y_matrix[:,v]); # get node v's caching variables (column vector)
        y[findall(i->(i<epsilon), y)] .= 0 # fix floating-point rounding errors by rounding all values in epsilon neighborhood of 0
        y[findall(i->(abs(i-1)<epsilon), y)] .= 1 # fix floating-point rounding errors by rounding all values in epsilon neighborhood of 1
        while !(all(isbinary, y)) # repeat as long as there are fractional values
            y_frac_pair_ind = findall(!isbinary, y)
            if length(y_frac_pair_ind)==1
                y_temp = deepcopy(y)
                display(y_frac_pair_ind)
                y_temp[y_frac_pair_ind] = 1
                if sum(y_temp)<=cache_capacity[v]
                    y[y_frac_pair_ind] = 1
                else
                    y[y_frac_pair_ind] = 0
                end
            else
                DO_best = Inf;
                y_best = zeros(Float64,M)

                for pair_ind_temp in collect(combinations(y_frac_pair_ind, 2))
                    y1 = y[pair_ind_temp][1]
                    y2 = y[pair_ind_temp][2]
                    y_temp = deepcopy(y)

                    # Case 1: Try rounding y1 to 0
                    if y1 + y2 < 1
                        y_temp[pair_ind_temp] = [0 y1+y2]
                        Y_temp = deepcopy(Y_matrix)
                        Y_temp[:,v] = y_temp
                        Y_temp = vcat(Y_temp...)
                        DO_temp = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ Gintegral[n](Y_temp) for n in 1:length(Gintegral) ])                        
                        if (DO_temp < DO_best)
                            DO_best = DO_temp
                            y_best = deepcopy(y_temp)
                        end
                    end

                    # Case 2: Try rounding y1 to 1
                    if y2 - (1 - y1) > 0
                        y_temp[pair_ind_temp] = [1 y2 - (1 - y1)]
                        Y_temp = deepcopy(Y_matrix)
                        Y_temp[:,v] = y_temp
                        Y_temp = vcat(Y_temp...)                       
                        DO_temp = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ Gintegral[n](Y_temp) for n in 1:length(Gintegral) ])                        
                        if (DO_temp < DO_best)
                            DO_best = DO_temp
                            y_best = deepcopy(y_temp)
                        end
                    end

                    # Case 3: Try rounding y2 to 0
                    if y1 + y2 < 1
                        y_temp[pair_ind_temp] = [y1+y2 0]
                        Y_temp = deepcopy(Y_matrix)
                        Y_temp[:,v] = y_temp
                        Y_temp = vcat(Y_temp...)                      
                        DO_temp = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ Gintegral[n](Y_temp) for n in 1:length(Gintegral) ])
                        if (DO_temp < DO_best)
                            DO_best = DO_temp
                            y_best = deepcopy(y_temp)
                        end
                    end

                    # Case 4: Try rounding y2 to 1
                    if y1 - (1 - y2) > 0
                        y_temp[pair_ind_temp] = [y1-(1-y2) 1]
                        Y_temp = deepcopy(Y_matrix)
                        Y_temp[:,v] = y_temp
                        Y_temp = vcat(Y_temp...)                     
                        DO_temp = sum([ F[m](S_opt) for m in 1:length(F) ] .* [ Gintegral[n](Y_temp) for n in 1:length(Gintegral) ])        
                        if (DO_temp < DO_best)
                            DO_best = DO_temp
                            y_best = deepcopy(y_temp)
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

function lruAdvance(time_slot::Int64, lru_timestamps::Array{Int64, 2}, lru_cache::BitArray{2}, reqs::requests, paths::Array{Array{Int64,1},2}, cache_capacity::Array{Int64,1})
    M = size(lru_cache, 1)
    V = size(lru_cache, 2)
    lru_temp = lru_timestamps .* lru_cache # last access timestamps of items still in caches
    lru_temp[ findall((i -> i<= 0), lru_temp) ] .= 1e8 # remove items not cached, if there are Inf (1e8) min values there's a bug possibly with initial values
    lru_temp = lru_temp[end:-1:1,1:1:end] # flip(lru_temp)
    evictions = argmin.([ lru_temp[1:end,i] for i in 1:V ]) # LRU items that are still in cache is next eviction, get indices from flipped vector so that least popular item will be given in cases of time collision
    evictions = (M+1) .- evictions; # get true indices

    # LRU update loop

    for (r,p) in enumerate(paths)
        plen = length(p)
        i = reqs.items[r]
        lambda_i_p = reqs.rates[r]
        for k=2:plen - 1 # Skip user node (start from k=2, CRUCIAL!)
            pk = p[k]
            lru_timestamps[i,pk] = time_slot;
            if lru_cache[i,pk]==0
                if sum(lru_cache[:,pk]) >= cache_capacity[pk]
                    if lru_cache[evictions[pk],pk] == 1
                        lru_cache[evictions[pk],pk] = 0
                    else
                        evict_temp = lru_timestamps[:,pk] .* lru_cache[:,pk]
                        evict_temp[ findall((i -> i<= 0), evict_temp) ] .= 1e8
                        evict_temp = evict_temp[end:-1:1] # flip(evict_temp)
                        eviction = (M+1) - argmin(evict_temp)
                        lru_cache[eviction,pk] = 0
                    end
                end
                lru_cache[i,pk] = 1
            else
                break
            end
        end
    end

    return lru_timestamps, lru_cache
end

function lfuAdvance(lfu_counts::Array{Int64, 2}, lfu_cache::BitArray{2}, reqs::requests, paths::Array{Array{Int64,1},2}, cache_capacity::Array{Int64,1})
    M = size(lfu_cache, 1)
    V = size(lfu_cache, 2)
    lfu_temp = lfu_counts .* lfu_cache # counts of items still in caches (counts will start at 1 instead of 0 to avoid miscalculation since we're ANDing)
    lfu_temp[ findall((i -> i<= 0), lfu_temp) ] .= 1e8 # remove items not cached, if there are Inf (1e8) min values there's a bug possibly with initial values
    lfu_temp = lfu_temp[end:-1:1,1:1:end] # flip(lfu_temp)
    evictions = argmin.([ lfu_temp[1:end,i] for i in 1:V ]) # LRU items that are still in cache is next eviction, get indices from flipped vector so that least popular item will be given in cases of time collision
    evictions = (M+1) .- evictions; # get true indices

    # LFU update loop

    for (r,p) in enumerate(paths)
        plen = length(p)
        i = reqs.items[r]
        lambda_i_p = reqs.rates[r]
        for k=2:plen - 1 # Skip user node (start from k=2, CRUCIAL!)
            pk = p[k]
            lfu_counts[i,pk] += 1;
            if lfu_cache[i,pk]==0
                if sum(lfu_cache[:,pk]) >= cache_capacity[pk]
                    if lfu_cache[evictions[pk],pk] == 1
                        lfu_cache[evictions[pk],pk] = 0
                    else
                        evict_temp = lfu_counts[:,pk] .* lfu_cache[:,pk]
                        evict_temp[ findall((i -> i<= 0), evict_temp) ] .= 1e8
                        evict_temp = evict_temp[end:-1:1] # flip(evict_temp)
                        eviction = (M+1) - argmin(evict_temp)
                        lfu_cache[eviction,pk] = 0
                    end
                end
                lfu_cache[i,pk] = 1
            else
                break
            end
        end
    end

    return lfu_counts, lfu_cache
end

function lrfuAdvance(time_slot::Int64, lrfu_scores::Array{Float64, 2}, lrfu_timestamps::Array{Int64, 2}, lrfu_cache::BitArray{2}, reqs::requests, paths::Array{Array{Int64,1},2}, cache_capacity::Array{Int64,1}, p::Float64 = 2.0, alpha::Float64 = 0.01)

    # Algorithm adapted from "LRFU: A Spectrum of Policies thatSubsumes the Least Recently Used andLeast Frequently Used Policies" by Donghee Lee et al. December 2001 IEEE Transactions on Computers Vol. 50 No. 12

    M = size(lrfu_cache, 1)
    V = size(lrfu_cache, 2)
    wf = (x -> (1/p)^(alpha*x))
    lrfu_temp = lrfu_scores .* lrfu_cache # counts of items still in caches (counts will start at 1 instead of 0 to avoid miscalculation since we're ANDing)
    lrfu_temp[ findall((i -> i<= 0), lrfu_temp) ] .= 1e8 # remove items not cached, if there are Inf (1e8) min values there's a bug possibly with initial values
    lrfu_temp = lrfu_temp[end:-1:1,1:1:end] # flip(lrfu_temp)
    evictions = argmin.([ lrfu_temp[1:end,i] for i in 1:V ]) # LRU items that are still in cache is next eviction, get indices from flipped vector so that least popular item will be given in cases of time collision
    evictions = (M+1) .- evictions; # get true indices

    # LRFU update loop

    for (r,p) in enumerate(paths)
        plen = length(p)
        i = reqs.items[r]
        lambda_i_p = reqs.rates[r]
        for k=2:plen - 1 # Skip user node (start from k=2, CRUCIAL!)
            pk = p[k]
            lrfu_scores[i,pk] = wf(0) + wf(time_slot - lrfu_timestamps[i,pk]) * lrfu_scores[i,pk]
            lrfu_timestamps[i,pk] = time_slot            
            if lrfu_cache[i,pk]==0
                if sum(lrfu_cache[:,pk]) >= cache_capacity[pk]
                    if lrfu_cache[evictions[pk],pk] == 1
                        lrfu_cache[evictions[pk],pk] = 0
                    else
                        evict_temp = lrfu_scores[:,pk] .* lrfu_cache[:,pk]
                        evict_temp[ findall((i -> i<= 0), evict_temp) ] .= 1e8
                        evict_temp = evict_temp[end:-1:1] # flip(evict_temp)
                        eviction = (M+1) - argmin(evict_temp)
                        lrfu_cache[eviction,pk] = 0
                    end
                end
                #lrfu_scores[i,pk] = wf(0) + wf(time_slot - lrfu_timestamps[i,pk]) * lrfu_scores[i,pk]
                #lrfu_timestamps[i,pk] = time_slot
                lrfu_cache[i,pk] = 1
            else
                break
            end
        end
    end

    return lrfu_scores, lrfu_timestamps, lrfu_cache
end

function fifoAdvance(fifo_queues::Array{Queue{Int64},1}, fifo_cache::BitArray{2}, reqs::requests, paths::Array{Array{Int64,1},2}, cache_capacity::Array{Int64,1})

    # FIFO update loop

    for (r,p) in enumerate(paths)
        plen = length(p)
        i = reqs.items[r]
        lambda_i_p = reqs.rates[r]
        for k=2:plen - 1 # Skip user node (start from k=2, CRUCIAL!)
            pk = p[k]
            q = fifo_queues[pk]
            if !in(i,collect(q)) # if item is not in cache
                if length(q) >= cache_capacity[pk]
                    dequeue!(q)
                end
                enqueue!(q,i)
            else
                break
            end
        end
    end

    fifo_cache[:,:] .= 0
    for v in 1:size(fifo_cache,2)
        fifo_cache[collect(fifo_queues[v]), v] .= 1
    end

    return fifo_queues, fifo_cache
end

function randAdvance(rand_cache::BitArray{2}, reqs::requests, paths::Array{Array{Int64,1},2}, cache_capacity::Array{Int64,1})
    M = size(rand_cache, 1)
    V = size(rand_cache, 2)
    
    # RANDOM update loop

    for (r,p) in enumerate(paths)
        plen = length(p)
        i = reqs.items[r]
        lambda_i_p = reqs.rates[r]
        for k=2:plen - 1 # Skip user node (start from k=2, CRUCIAL!)
            pk = p[k]
            if rand_cache[i,pk]==0
                if sum(rand_cache[:,pk]) >= cache_capacity[pk]
                    eviction = rand(findall( (i -> i==1), rand_cache[:,pk] ))
                    rand_cache[eviction,pk] = 0
                end
                rand_cache[i,pk] = 1
            else
                break
            end
        end
    end

    return rand_cache
end

function isbinary(x)
    if (x==0 || x==1)
        return true
    else
        return false
    end
end