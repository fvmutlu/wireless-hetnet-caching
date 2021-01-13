include("basic_functions.jl")
using Convex, SCS, Random, Distributions, Dates, Combinatorics

function cellGraph(V,SC,R_cell)
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
    sc_nodes = zeros(Int64, 1, SC)
    sc_count = 0
    # Assign random node to be first SC
    while (sc_count == 0)
        rand_node = rand(2:V)
        if sqrt(V_pos[rand_node, 1]^2 + V_pos[rand_node, 2]^2) > R_sc
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

    # u_nodes = findall(i->!(i in sc_nodes),2:V) # Create array with U node IDs

    # G = SimpleWeightedDiGraph(V+1)

    return 0
end

function projOpt(S_step_t, P_min, P_max, Y_step_t, C, cache_capacity)
    # Minimum norm subproblem for S projection
    dim_S = size(S_step_t, 1) # length of power vector
    S_proj_t = Variable(dim_S) # problem variable, a column vector (Convex.jl)
    problem = minimize(norm(S_proj_t - S_step_t),[S_proj_t >= P_min, ones(Int64, 1, dim_S)*S_proj_t <= P_max]) # problem definition (Convex.jl)
    solve!(problem, SCS.Optimizer) # use SCS solver (Convex.jl, SCS.jl)

    # Minimum norm subproblem for Y projection
    dim_Y = size(Y_step_t,1)
    Y_proj_t = Variable(dim_Y)
    problem = minimize(norm(Y_proj_t - Y_step_t),[Y_proj_t >= 0, Y_proj_t <= 1, C*Y_proj_t <= cache_capacity])
    solve!(problem, SCS.Optimizer)
    
    return [evaluate(S_proj_t), evaluate(Y_proj_t)]
end

function randomInitPoint(dim_S, dim_Y, P_min, P_max, C, cache_capacity, weight)
    Random.seed!(Dates.value(Dates.now())) # set the seed with current system time

    # Randomized initial point for power vector
    mean = (P_max + P_min)/2 + (P_max + P_min)*weight/2
    var = (P_max + P_min)/10
    pd = Normal(mean, var)
    S_0 = rand(pd, dim_S)

    # for caching vector
    mean = 0.5 + 0.5*weight/2
    var = 0.1
    pd = Normal(mean, var)
    Y_0 = rand(pd,dim_Y)

    return projOpt(S_0, P_min, P_max, Y_0, C, cache_capacity)
end

function pipageRound(F, Gintegral, Y_opt, S_opt, M, V, cache_capacity)
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
                DO_best = 9999999999.9;
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
                        Y_temp = reshape(Y_temp, (M*V, 1))                        
                        DO_temp = dot(F(S_opt),Gintegral(Y_temp))                        
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
                        Y_temp = reshape(Y_temp, (M*V,1))                        
                        DO_temp = dot(F(S_opt),Gintegral(Y_temp))                        
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
                        Y_temp = reshape(Y_temp, (M*V,1))                        
                        DO_temp = dot(F(S_opt),Gintegral(Y_temp))
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
                        Y_temp = reshape(Y_temp, (M*V,1))                        
                        DO_temp = dot(F(S_opt),Gintegral(Y_temp))                        
                        if (DO_temp < DO_best)
                            DO_best = DO_temp
                            y_best = y_temp
                        end
                    end
                end
                y = round.(y_best); # If no cases had better than the current objective, this will just pick the best result out of all trials.
                # We round because the results from projection can be arbitrarily close to 1 or 0 while they should be exactly 1 or 0 (e.g. 0.9999999875 or 4.57987363e-11), I suspect this is due to inner workings of Convex.jl
            end
        end
        Y_matrix[:,v] = y
    end
    return reshape(Y_matrix, (M*V,1))
end
