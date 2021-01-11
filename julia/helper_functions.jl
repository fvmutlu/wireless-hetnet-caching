include("basic_functions.jl")
using Convex, SCS, Random, Distributions, Dates, Combinatorics

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
    
    return [S_proj_t, Y_proj_t]
end

function randomInitPoint(dim_S, dim_Y, P_min, P_max, C, cache_capacity, weight)
    Random.seed!(Dates.value(Dates.now())) # set the seed with current system time

    # Randomized initial point for power vector
    mean = (P_max + P_min)/2 + (P_max + P_min)*weight/2
    var = (P_max + P_min)/10
    pd = Normal(mean, var)
    S_0 = rand(pd, dim_S)

    # for caching vector
    mean = 0.5 + 0.5*weight/2;
    var = 0.1;
    pd = Normal(mean, var)
    Y_0 = rand(pd,dim_Y);

    return projOpt(S_0, P_min, P_max, Y_0, C, cache_capacity)
end

function pipageRound(F, Gintegral, Y_opt, S_opt, M, V, cache_capacity)
    Y_matrix = reshape(Y_opt, (M, V)) # put the caching vector into matrix form
    epsilon = 1e-3
    for v in 1:V # repeat for all nodes 
        y = Y_matrix(:,v); # get node v's caching variables (column vector)
        y[findall(i->(i<epsilon), y)] .= 0 # fix floating-point rounding errors by rounding all values in epsilon neighborhood of 0
        y[findall(i->(abs(i-1)<epsilon), y)] .= 1 # fix floating-point rounding errors by rounding all values in epsilon neighborhood of 1
        while !(all(isbinary, y)) # repeat as long as there are fractional values
            y_frac_pair_ind = findall(!isbinary, y)
            if(length(y_frac_pair_ind)==1)
                y_temp = y
                y_temp[y_frac_pair_ind] = 1
                if(sum(y_temp)<=cache_capacity[v])
                    y[y_frac_pair_ind] = 1
                else
                    y[y_frac_pair_ind] = 0
                end
            else
                DO_best = Inf;
                for pair_ind_temp in collect(combinations(y_frac_pair_ind, 2))
                    y_frac_pair = y[pair_ind_temp]
                    y1 = y_frac_pair[1]
                    y2 = y_frac_pair[2]
                    y_temp = y

                    # Case 1: Try rounding y1 to 0
                    if(y1 + y2 < 1)
                        y_temp[pair_ind_temp] = [0 y1+y2];
                        Y_temp = Y_matrix;
                        Y_temp[:,v] = y_temp;
                        Y_temp = reshape(Y_temp, (M*V, 1));
                        DO_temp = F(S_opt')*transpose(Gintegral(Y_temp'));
                        if (DO_temp < DO_best)
                            DO_best = DO_temp;
                            y_best = y_temp;
                        end
                    end
                    # Case 2: Try rounding y1 to 1
                    if(y2-(1-y1)>0)
                        y_temp[pair_ind_temp] = [1 y2-(1-y1)];
                        Y_temp = Y_matrix;
                        Y_temp[:,v] = y_temp;
                        Y_temp = reshape(Y_temp, (M*V,1));
                        DO_temp = F(S_opt')*transpose(Gintegral(Y_temp'));
                        if (DO_temp < DO_best)
                            DO_best = DO_temp;
                            y_best = y_temp;
                        end
                    end
                    # Case 3: Try rounding y2 to 0
                    if(y1+y2<1)
                        y_temp[pair_ind_temp] = [y1+y2 0];
                        Y_temp = Y_matrix;
                        Y_temp[:,v] = y_temp;
                        Y_temp = reshape(Y_temp, (M*V,1));
                        DO_temp = F(S_opt')*transpose(Gintegral(Y_temp'));
                        if (DO_temp < DO_best)
                            DO_best = DO_temp;
                            y_best = y_temp;
                        end
                    end
                    # Case 4: Try rounding y2 to 1
                    if(y1-(1-y2)>0)
                        y_temp[pair_ind_temp] = [y1-(1-y2) 1];
                        Y_temp = Y_matrix;
                        Y_temp[:,v] = y_temp;
                        Y_temp = reshape(Y_temp, (M*V,1));
                        DO_temp = F(S_opt')*transpose(Gintegral(Y_temp'));
                        if (DO_temp < DO_best)
                            DO_best = DO_temp;
                            y_best = y_temp;
                        end
                    end
                end
                y = y_best; # If no cases had better than the current objective, this will just pick the best result out of all trials
            end
        end
        Y_matrix[:,v] = y;
    end
    X = reshape(Y_matrix, (M*V,1));
