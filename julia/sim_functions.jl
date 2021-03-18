include("methods.jl")
using Printf, Plots

mutable struct other_parameters
    M::Int64 # Total number of items in the catalog
    pd::Array{Float64,1} # Probability distribution for requests
    numof_requests::Int64 # Total number of requests across all users (NAMED THIS WAY FOR FUTURE CONSIDERATION, FOR CURRENT CASE THIS IS ALWAYS EQUAL TO NUMBER OF USERS) 
    D_bh_mc::Float64 # MC-BH wireline link delay (LEFT HERE FOR POSSIBLE CUSTOM ADJUSTMENT, CURRENTLY CALCULATED VIA HEURISTICS USING POWER CONSTRAINTS AND NETWORK PARAMETERS)
    D_bh_sc::Float64 # SC-BH wireline link delay, same for all SCs (LEFT HERE FOR POSSIBLE CUSTOM ADJUSTMENT, CURRENTLY CALCULATED VIA HEURISTICS USING POWER CONSTRAINTS AND NETWORK PARAMETERS)
    numof_initpoints::Int64 # Number of initial points to run the problem with
end

mutable struct network_parameters
    V::Int64 # Total number of nodes (Macro cell + small cells + users)
    SC::Int64 # Number of small cells (SCs)
    R_cell::Float64 # Radius of cell
    pathloss_exponent::Float64 # Self explanatory
    C_bh_mc::Float64 # MC-BH wireline link cost
    C_bh_sc::Float64 # MC-SC wireline link cost (same for all SCs)
    noise::Float64 # Average power of noise in the network (same everywhere)
end

mutable struct constraint_parameters
    P_min::Float64 # Minimum allowed power per transmitter node (MC and SCs) across all its transmissions
    P_max::Float64 # Maximum allowed power (either total across all transmitters or per transmitter)
    c_mc::Int64 # Cache capacity in number of items of MC
    c_sc::Int64 # Cache capacity in number of items of SC
end

mutable struct problem_components
    V_pos::Array{Float64,2}
    netgraph::network_graph
    reqs::requests
    funcs
    consts::constraints
    SY_0::Array{Tuple{Array{Float64,1},Array{Float64,1}},1}
end

function readConfig(input = "/home/volkan/opt-caching-power/julia/config.txt")
    cfg = split.(readlines(input), " # ")
    (V, SC, M, c_mc, c_sc, pd_type, numof_initpoints) = parse.(Int64, [ cfg[i][1] for i in 1:7 ])
    (P_min, P_max, pathloss_exponent, noise, R_cell, pd_param) = parse.(Float64, [ cfg[i][1] for i in 8:13 ])
    
    U = V - SC - 1
    C_bh_mc = (R_cell/2)^pathloss_exponent # Cost at the wireline edge between backhaul to macro cell (calculated completely heuristically, there could be smarter way of determining this)
    C_bh_sc = 2 * C_bh_mc # Cost at the wireline edge between backhaul and any of the small cells (again, heuristic)
    D_bh_mc = 1 / log( 1 + ((1/C_bh_mc)*P_max)/noise ) # Delay of data retrieval from backhaul to macro cell (again, computed heuristically using the wireline edge costs)
    D_bh_sc = 1 / log( 1 + ((1/C_bh_sc)*P_max*0.8)/noise ) # Delay of data retrieval from backhaul to any of the small cells (again, computed heuristically using the wireline edge costs)
    
    if pd_type == 0
        pd = (1 ./ (1:M).^ pd_param) / sum(1 ./ (1:M).^ pd_param)
    else # have different distributions here possibly?
        pd = (1 ./ (1:M).^ pd_param) / sum(1 ./ (1:M).^ pd_param)
    end
    
    params = other_parameters(M, pd, U, D_bh_mc, D_bh_sc, numof_initpoints) # numof_requests = U FOR CURRENT CASE
    netparams = network_parameters(V, SC, R_cell, pathloss_exponent, C_bh_mc, C_bh_sc, noise)
    constparams = constraint_parameters(P_min, P_max, c_mc, c_sc)

    return params, netparams, constparams
end

function newProblem(params::other_parameters = params, netparams::network_parameters = netparams, constparams::constraint_parameters = constparams)
    V_pos = cellDist(netparams.V, netparams.SC, netparams.R_cell)

    netgraph = makeGraph(V_pos, netparams.pathloss_exponent, netparams.C_bh_mc, netparams.C_bh_sc)

    reqs = randomRequests(params.pd, params.numof_requests, 1.0)

    funcs = funcSetup(netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)

    consts = makeConsts(netparams.V, params.M, constparams.c_mc, constparams.c_sc, findall(i -> i == 1, V_pos[:,3]), constparams.P_min, constparams.P_max)

    weight = (i, N) -> (N != 1) * (( i - ( (N + 1) / 2) ) / (N - 1))
    SY_0 = [ randomInitPoint(size(netgraph.edges,1), netparams.V*params.M, weight(i, params.numof_initpoints), consts) for i in 1:params.numof_initpoints ]

    #return V_pos, netgraph, reqs, funcs, consts, SY_0
    return problem_components(V_pos, netgraph, reqs, funcs, consts, SY_0)
end

## Run optimization over different initial points

function runSim(V::Int64 = netparams.V, M::Int64 = params.M, probcomps::problem_components = probcomps)
    funcs = probcomps.funcs
    consts = probcomps.consts
    SY_0 = probcomps.SY_0
    #= println(" -- SUB -- ")
    (D_opt, S_opt, Y_opt) = @time subMethod(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, M, V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0) =#

    #= println(" -- SUB MULT -- ")
    (D_opt, S_opt, Y_opt) = @time subMethodMult(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, M, V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0) =#

    println(" -- ALT --")
    (D_opt, S_opt, Y_opt) = @time altMethod(SY_0[1], funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, M, V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0)

    #= println(" -- ALT MULT -- ")
    (D_opt, S_opt, Y_opt) = @time altMethodMult(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, M, V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0) =#
end

function incScSim(inc_count::Int64, params::other_parameters = params, netparams::network_parameters = netparams, constparams::constraint_parameters = constparams, probcomps::problem_components = probcomps)
    results = zeros(Float64, inc_count+1)

    #V_pos, netgraph, reqs, funcs, consts, SY_0 = newProblem(params, netparams, constparams)
    V_pos = probcomps.V_pos
    netgraph = probcomps.netgraph
    reqs = probcomps.reqs
    funcs = probcomps.funcs
    consts = probcomps.consts
    SY_0 = probcomps.SY_0[1]
    #SY_0 = randomInitPoint(size(netgraph.edges,1), netparams.V*params.M, 0, consts)

    (D_opt, S_opt, Y_opt) = altMethod(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, netparams.V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    results[1] = D_0;

    for i in 2:inc_count+1
        netparams.V += 1
        netparams.SC += 1
        V_pos = addSC(V_pos, netparams.R_cell)
        netgraph = makeGraph(V_pos, netparams.pathloss_exponent, netparams.C_bh_mc, netparams.C_bh_sc)
        funcs = funcSetup(netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        consts = makeConsts(netparams.V, params.M, constparams.c_mc, constparams.c_sc, findall(i -> i == 1, V_pos[:,3]), constparams.P_min, constparams.P_max)
        SY_0 = randomInitPoint(size(netgraph.edges,1), netparams.V*params.M, 0, consts)

        (D_opt, S_opt, Y_opt) = altMethod(SY_0, funcs, consts)
        X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, netparams.V, consts.cache_capacity)
        D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
        results[i] = D_0;
    end

    return plot((netparams.SC - inc_count):netparams.SC, results, title="Delay with increasing # of SCs", xlabel="# of SCs", ylabel="Delay")
end

function incPwrSim(pwr_lim::Float64, pwr_inc::Float64, params::other_parameters = params, netparams::network_parameters = netparams, constparams::constraint_parameters = constparams)
    pwr_range = range(constparams.P_max, step = pwr_inc, stop = pwr_lim)
    results = zeros(Float64, length(pwr_range))

    V_pos, netgraph, reqs, funcs, consts, SY_0 = newProblem(params, netparams, constparams)
    SY_0 = randomInitPoint(size(netgraph.edges,1), netparams.V*params.M, 0, consts)

    for (i, P_max) in enumerate(pwr_range)
        consts = makeConsts(netparams.V, params.M, constparams.c_mc, constparams.c_sc, findall(n -> n == 1, V_pos[:,3]), constparams.P_min, P_max)
        (D_opt, S_opt, Y_opt) = altMethod(SY_0, funcs, consts)
        X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, netparams.V, consts.cache_capacity)
        D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
        results[i] = D_0;
    end

    return plot(pwr_range, results, title="Delay with increasing Pmax", xlabel="Pmax", ylabel="Delay")
end

function lruSim(T::Int64, params::other_parameters = params, netparams::network_parameters = netparams, probcomps::problem_components = probcomps)
    M = params.M
    V = netparams.V
    cache_capacity = probcomps.consts.cache_capacity
    paths = probcomps.netgraph.paths
    F = probcomps.funcs.F
    Gintegral = probcomps.funcs.Gintegral

    lru_timestamps = ones(Int64, M,V)
    lru_cache = BitArray(undef, M,V)
    lru_cache[:,:] .= 0
    for v in 1:V
        lru_cache[1:cache_capacity[v],v] .= 1
    end
    S_0 = (probcomps.consts.P_max / size(probcomps.netgraph.edges,1)) * ones(Float64, size(probcomps.netgraph.edges,1))

    X_0 = deepcopy(reshape(lru_cache,M*V))
    D_lru = ThreadsX.sum( ThreadsX.collect(F[m](S_0) for m in 1:length(F)) .* ThreadsX.collect(Gintegral[n](X_0) for n in 1:length(Gintegral)) )
    t = 2
    t, lru_timestamps, lru_cache = lruAdvance(t, lru_timestamps, lru_cache, probcomps.reqs, paths, cache_capacity)
    while t <= T+1
        X_t = deepcopy(reshape(lru_cache,M*V))
        reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D_lru += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_t) for n in 1:length(funcs.Gintegral)) )
        t, lru_timestamps, lru_cache = lruAdvance(t, lru_timestamps, lru_cache, reqs, paths, cache_capacity)        
    end
    
    println(" -- LRU --")
    @printf("Delay: %.2f\n", D_lru/(T))
    return D_lru/(T)
end

function lfuSim(T::Int64, params::other_parameters = params, netparams::network_parameters = netparams, probcomps::problem_components = probcomps)
    M = params.M
    V = netparams.V
    cache_capacity = probcomps.consts.cache_capacity
    paths = probcomps.netgraph.paths
    F = probcomps.funcs.F
    Gintegral = probcomps.funcs.Gintegral

    lfu_counts = ones(Int64, M,V)
    lfu_cache = BitArray(undef, M,V)
    lfu_cache[:,:] .= 0
    for v in 1:V
        lfu_cache[1:cache_capacity[v],v] .= 1
    end
    S_0 = (probcomps.consts.P_max / size(probcomps.netgraph.edges,1)) * ones(Float64, size(probcomps.netgraph.edges,1))

    X_0 = deepcopy(reshape(lfu_cache,M*V))
    D_lfu = ThreadsX.sum( ThreadsX.collect(F[m](S_0) for m in 1:length(F)) .* ThreadsX.collect(Gintegral[n](X_0) for n in 1:length(Gintegral)) )
    t = 2
    t, lfu_counts, lfu_cache = lfuAdvance(t, lfu_counts, lfu_cache, probcomps.reqs, paths, cache_capacity)
    while t <= T+1
        X_t = deepcopy(reshape(lfu_cache,M*V))
        reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D_lfu += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_t) for n in 1:length(funcs.Gintegral)) )
        t, lfu_counts, lfu_cache = lfuAdvance(t, lfu_counts, lfu_cache, reqs, paths, cache_capacity)        
    end
    
    println(" -- LFU --")
    @printf("Delay: %.2f\n", D_lfu/(T))
    return D_lfu/(T)
end

function fifoSim(T::Int64, params::other_parameters = params, netparams::network_parameters = netparams, probcomps::problem_components = probcomps)
    M = params.M
    V = netparams.V
    cache_capacity = probcomps.consts.cache_capacity
    paths = probcomps.netgraph.paths
    F = probcomps.funcs.F
    Gintegral = probcomps.funcs.Gintegral

    fifo_queues = collect( [ Queue{Int64}() for i in 1:V] )
    fifo_cache = BitArray(undef, M,V)
    fifo_cache[:,:] .= 0
    for v in 1:V
        for i in 1:cache_capacity[v]
            enqueue!(fifo_queues[v],i)
        end
        fifo_cache[1:cache_capacity[v],v] .= 1
    end
    S_0 = (probcomps.consts.P_max / size(probcomps.netgraph.edges,1)) * ones(Float64, size(probcomps.netgraph.edges,1))

    X_0 = deepcopy(reshape(fifo_cache,M*V))
    D_fifo = ThreadsX.sum( ThreadsX.collect(F[m](S_0) for m in 1:length(F)) .* ThreadsX.collect(Gintegral[n](X_0) for n in 1:length(Gintegral)) )
    t = 2
    t, fifo_queues, fifo_cache = fifoAdvance(t, fifo_queues, fifo_cache, probcomps.reqs, paths, cache_capacity)
    while t <= T+1
        X_t = deepcopy(reshape(fifo_cache,M*V))
        reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D_fifo += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_t) for n in 1:length(funcs.Gintegral)) )
        t, fifo_queues, fifo_cache = fifoAdvance(t, fifo_queues, fifo_cache, reqs, paths, cache_capacity)        
    end
    
    println(" -- FIFO --")
    @printf("Delay: %.2f\n", D_fifo/(T))
    return D_fifo/(T)
end

function randSim(T::Int64, params::other_parameters = params, netparams::network_parameters = netparams, probcomps::problem_components = probcomps)
    M = params.M
    V = netparams.V
    cache_capacity = probcomps.consts.cache_capacity
    paths = probcomps.netgraph.paths
    F = probcomps.funcs.F
    Gintegral = probcomps.funcs.Gintegral

    rand_cache = BitArray(undef, M,V)
    rand_cache[:,:] .= 0
    for v in 1:V
        rand_cache[1:cache_capacity[v],v] .= 1
    end
    S_0 = (probcomps.consts.P_max / size(probcomps.netgraph.edges,1)) * ones(Float64, size(probcomps.netgraph.edges,1))

    X_0 = deepcopy(reshape(rand_cache,M*V))
    D_rand = ThreadsX.sum( ThreadsX.collect(F[m](S_0) for m in 1:length(F)) .* ThreadsX.collect(Gintegral[n](X_0) for n in 1:length(Gintegral)) )
    t = 2
    t, rand_cache = randAdvance(t, rand_cache, probcomps.reqs, paths, cache_capacity)
    while t <= T+1
        X_t = deepcopy(reshape(rand_cache,M*V))
        reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D_rand += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_t) for n in 1:length(funcs.Gintegral)) )
        t, rand_cache = randAdvance(t, rand_cache, reqs, paths, cache_capacity)        
    end
    
    println(" -- RANDOM --")
    @printf("Delay: %.2f\n", D_rand/(T))
    return D_rand/(T)
end

function polruSim(T::Int64, params::other_parameters = params, netparams::network_parameters = netparams, probcomps::problem_components = probcomps)
    M = params.M
    V = netparams.V
    cache_capacity = probcomps.consts.cache_capacity
    paths = probcomps.netgraph.paths
    F = probcomps.funcs.F
    Gintegral = probcomps.funcs.Gintegral

    lru_timestamps = ones(Int64, M,V)
    lru_cache = BitArray(undef, M,V)
    lru_cache[:,:] .= 0
    for v in 1:V
        lru_cache[1:cache_capacity[v],v] .= 1
    end
    S_0 = (probcomps.consts.P_max / size(probcomps.netgraph.edges,1)) * ones(Float64, size(probcomps.netgraph.edges,1))

    X_0 = deepcopy(reshape(lru_cache,M*V))
    D_polru, S_t = pwrOnly(S_0, X_0, probcomps.funcs, probcomps.consts)
    t = 2
    t, lru_timestamps, lru_cache = lruAdvance(t, lru_timestamps, lru_cache, probcomps.reqs, paths, cache_capacity)
    while t <= T+1
        X_t = deepcopy(reshape(lru_cache,M*V))
        reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D, S_t = pwrOnly(S_t, X_t, probcomps.funcs, probcomps.consts)
        D_polru += D
        t, lru_timestamps, lru_cache = lruAdvance(t, lru_timestamps, lru_cache, reqs, paths, cache_capacity)
    end
    
    println(" -- POLRU --")
    @printf("Delay: %.2f\n", D_polru/(T))
end

function incScCombSim(inc_count::Int64, params::other_parameters = params, netparams::network_parameters = netparams, constparams::constraint_parameters = constparams, probcomps::problem_components = probcomps)
    results = zeros(Float64, inc_count+1)
    lru_results = zeros(Float64, inc_count+1)
    lfu_results = zeros(Float64, inc_count+1)
    fifo_results = zeros(Float64, inc_count+1)
    rand_results = zeros(Float64, inc_count+1)

    V_pos = probcomps.V_pos
    netgraph = probcomps.netgraph
    reqs = probcomps.reqs
    funcs = probcomps.funcs
    consts = probcomps.consts
    SY_0 = probcomps.SY_0[1]

    (D_opt, S_opt, Y_opt) = altMethod(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, netparams.V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    results[1] = D_0;
    lru_results[1] = lruSim(200)
    lfu_results[1] = lfuSim(200)
    fifo_results[1] = fifoSim(200)
    rand_results[1] = randSim(200)

    for i in 2:inc_count+1
        netparams.V += 1
        netparams.SC += 1
        V_pos = addSC(V_pos, netparams.R_cell)
        netgraph = makeGraph(V_pos, netparams.pathloss_exponent, netparams.C_bh_mc, netparams.C_bh_sc)
        funcs = funcSetup(netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        consts = makeConsts(netparams.V, params.M, constparams.c_mc, constparams.c_sc, findall(i -> i == 1, V_pos[:,3]), constparams.P_min, constparams.P_max)
        SY_0 = randomInitPoint(size(netgraph.edges,1), netparams.V*params.M, 0, consts)

        (D_opt, S_opt, Y_opt) = altMethod(SY_0, funcs, consts)
        X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, netparams.V, consts.cache_capacity)
        D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
        results[i] = D_0;

        newprob = problem_components(V_pos, netgraph, reqs, funcs, consts, probcomps.SY_0)
        lru_results[i] = lruSim(200, params, netparams, newprob)
        lfu_results[i] = lfuSim(200, params, netparams, newprob)
        fifo_results[i] = fifoSim(200, params, netparams, newprob)
        rand_results[i] = randSim(200, params, netparams, newprob)
    end

    display(hcat(results,lru_results,lfu_results,fifo_results,rand_results))

    return plot((netparams.SC - inc_count):netparams.SC, hcat(results,lru_results,lfu_results,fifo_results,rand_results), title="Delay with increasing # of SCs", label = ["ALT" "LRU" "LFU" "FIFO" "RANDOM"], xlabel="# of SCs", ylabel="Delay")
end