include("methods.jl")
using Printf, Plots

import Base.Threads.@spawn

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
    C_bh_mc = max((R_cell/2),1)^pathloss_exponent # Cost at the wireline edge between backhaul to macro cell (calculated completely heuristically, there could be smarter way of determining this)
    C_bh_sc = 5 * C_bh_mc # Cost at the wireline edge between backhaul and any of the small cells (again, heuristic)
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

    #consts = makeConsts(netparams.V, params.M, constparams.c_mc, constparams.c_sc, findall(i -> i == 1, V_pos[:,3]), constparams.P_min, constparams.P_max)
    consts = makeConsts(netparams.V, params.M, findall(i -> i == 1, V_pos[:,3]), netgraph.edges, constparams.c_mc, constparams.c_sc, constparams.P_min, constparams.P_max)

    weight = (i, N) -> (N != 1) * (( i - ( (N + 1) / 2) ) / (N - 1))
    SY_0 = [ randomInitPoint(size(netgraph.edges,1), netparams.V*params.M, weight(i, params.numof_initpoints), consts) for i in 1:params.numof_initpoints ]

    #return V_pos, netgraph, reqs, funcs, consts, SY_0
    return problem_components(V_pos, netgraph, reqs, funcs, consts, SY_0)
end

## Run optimization over different initial points

function runSim(params::other_parameters = params, netparams::network_parameters = netparams, constparams::constraint_parameters = constparams, probcomps::problem_components = probcomps)
    open("/home/volkan/Repos/wireless-hetnet-caching/julia/convergenceresults.txt","a") do io
        println(io, "--------")
        println(io, "--params--")
        println(io, params)
        println(io, "--netparams--")
        println(io, netparams)
        println(io, "--constparams--")
        println(io, constparams)
    end

    funcs = probcomps.funcs
    consts = probcomps.consts
    #SY_0 = probcomps.SY_0
    #S_0 = (consts.P_max / size(probcomps.netgraph.edges,1)) * ones(Float64, size(probcomps.netgraph.edges,1)) # total
    S_0 = (consts.P_min) * ones(Float64, size(probcomps.netgraph.edges,1)) # per node
    Y_0 = zeros(Float64, netparams.V*params.M)
    SY_0 = (S_0, Y_0)

    println(" -- SUB -- ")
    (D_sub, S_sub, Y_sub, D_sub_arr, cputime_sub_arr) = subMethod(SY_0, funcs, consts)
    X_sub = pipageRounding(funcs.F, funcs.Gintegral, S_sub, Y_sub, params.M, netparams.V, consts.cache_capacity)
    D_0_sub = sum([ funcs.F[m](S_sub) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_sub) for n in 1:length(funcs.G) ])
    @printf("Delay: %.2f, Not Rounded: %.2f\n", D_0_sub, D_sub)
    D_0_sub, S_sub = pwrOnly(S_sub, X_sub, funcs, consts)
    @printf("One more pwr: %.2f\n",D_0_sub)

    println(" -- ALT -- ")
    (D_alt, S_alt, Y_alt, D_alt_arr, cputime_alt_arr) = altMethod(SY_0, funcs, consts)
    X_alt = pipageRounding(funcs.F, funcs.Gintegral, S_alt, Y_alt, params.M, netparams.V, consts.cache_capacity)
    D_0_alt = sum([ funcs.F[m](S_alt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_alt) for n in 1:length(funcs.G) ])
    @printf("Delay: %.2f, Not Rounded: %.2f\n", D_0_alt, D_alt)
    D_0_alt, S_alt = pwrOnly(S_alt, X_alt, funcs, consts)
    @printf("One more pwr: %.2f\n",D_0_alt)

    open("/home/volkan/Repos/wireless-hetnet-caching/julia/convergenceresults.txt","a") do io
        println(io,"-- SUB --")
        println(io,D_sub_arr)
        println(io,cputime_sub_arr)
        println(io,"-- ALT --")
        println(io,D_alt_arr)
        println(io,cputime_alt_arr)
    end

    #= println(" -- SUB MULT -- ")
    (D_opt, S_opt, Y_opt) = @time subMethodMult(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, M, V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0) =#

    #= println(" -- ALT --")
    (D_opt, S_opt, Y_opt) = @time altMethod(SY_0[1], funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, M, V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0) =#

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

function incPwrSim(pwr_lim::Float64, pwr_inc::Float64, T::Int64, params::other_parameters = params, netparams::network_parameters = netparams, constparams::constraint_parameters = constparams, probcomps::problem_components = probcomps)
    open("/home/volkan/opt-caching-power/julia/pwrresults.txt","a") do io
        println(io, "--------")
        println(io, "--params--")
        println(io, params)
        println(io, "--netparams--")
        println(io, netparams)
        println(io, "--constparams--")
        println(io, constparams)
    end

    pwr_range = range(constparams.P_max, step = pwr_inc, stop = pwr_lim)
    results = zeros(Float64, length(pwr_range))
    sub_results = zeros(Float64, length(pwr_range))
    lru_results = zeros(Float64, length(pwr_range))
    lfu_results = zeros(Float64, length(pwr_range))
    fifo_results = zeros(Float64, length(pwr_range))

    reqs, items_large, rates_large = averageRequests(T+1, 5, params.pd, params.numof_requests, 0.5)
    funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
    #SY_0 = randomInitPoint(size(probcomps.netgraph.edges,1), netparams.V*params.M, 0, probcomps.consts)
    #S_0 = (probcomps.consts.P_max / size(probcomps.netgraph.edges,1)) * ones(Float64, size(probcomps.netgraph.edges,1)) # total
    S_0 = (probcomps.consts.P_min) * ones(Float64, size(probcomps.netgraph.edges,1)) # per node
    Y_0 = zeros(Float64, netparams.V*params.M)
    SY_0 = (S_0, Y_0)

    for (i, P_max) in enumerate(pwr_range)
        consts = makeConsts(netparams.V, params.M, findall(n -> n == 1, probcomps.V_pos[:,3]), probcomps.netgraph.edges, constparams.c_mc, constparams.c_sc, constparams.P_min, P_max)

        #weight = (i, N) -> (N != 1) * (( i - ( (N + 1) / 2) ) / (N - 1))
        #SY_0 = [ randomInitPoint(size(probcomps.netgraph.edges,1), netparams.V*params.M, weight(i, params.numof_initpoints), consts) for i in 1:params.numof_initpoints ]

        (D_opt, S_opt, Y_opt, _, _) = altMethod(SY_0, funcs, consts) # Since we're strictly increasing Pmax, we can reuse the same initial points
        X_opt = pipageRounding(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, netparams.V, consts.cache_capacity)
        D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.Gintegral) ])
        @printf("--/ Pmax = %f /--\n",P_max)
        println("--ALT--")
        @printf("Delay: %.2f, Not Rounded: %.2f\n", D_0, D_opt)
        D_0, S_opt = pwrOnly(S_opt, X_opt, funcs, consts)
        @printf("One more pwr: %.2f\n",D_0)
        results[i] = D_0;

        #(D_sub, S_sub, Y_sub) = subMethodMult(SY_0, funcs, consts)
        (D_sub, S_sub, Y_sub, _, _) = subMethod(SY_0, funcs, consts)
        X_sub = pipageRounding(funcs.F, funcs.Gintegral, S_sub, Y_sub, params.M, netparams.V, consts.cache_capacity)
        D_0_sub = sum([ funcs.F[m](S_sub) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_sub) for n in 1:length(funcs.Gintegral) ])
        println("--SUB--")
        @printf("Delay: %.2f, Not Rounded: %.2f\n", D_0_sub, D_sub)
        D_0_sub, S_sub = pwrOnly(S_sub, X_sub, funcs, consts)
        @printf("One more pwr: %.2f\n",D_0_sub)
        sub_results[i] = D_0_sub;

        newprob = problem_components(probcomps.V_pos, probcomps.netgraph, reqs, funcs, consts, probcomps.SY_0)
        lru_results[i], lfu_results[i], fifo_results[i] = baselinePoSim(T, items_large, rates_large, params, netparams, newprob)
    end

    open("/home/volkan/opt-caching-power/julia/pwrresults.txt","a") do io
        println(io, "--ALT--")
        println(io, results)
        println(io, "--SUB--")
        println(io, sub_results)
        println(io, "--POLRU--")
        println(io, lru_results)
        println(io, "--POLFU--")
        println(io, lfu_results)
        println(io, "--POFIFO--")
        println(io, fifo_results)
    end

    return plot(pwr_range, hcat(results,sub_results,lru_results,lfu_results,fifo_results), title="Delay with increasing Pmax", label = ["ALT" "SUB" "LRU" "LFU" "FIFO"], xlabel="Pmax", ylabel="Delay")
end

function incCacheSim(cache_lim::Int64, T::Int64, params::other_parameters = params, netparams::network_parameters = netparams, constparams::constraint_parameters = constparams, probcomps::problem_components = probcomps)
    open("/home/volkan/opt-caching-power/julia/cacheresults.txt","a") do io
        println(io, "--------")
        println(io, "--params--")
        println(io, params)
        println(io, "--netparams--")
        println(io, netparams)
        println(io, "--constparams--")
        println(io, constparams)
    end

    cache_range = range(1, step = 1, stop = cache_lim)
    results = zeros(Float64, length(cache_range))
    sub_results = zeros(Float64, length(cache_range))
    lru_results = zeros(Float64, length(cache_range))
    lfu_results = zeros(Float64, length(cache_range))
    fifo_results = zeros(Float64, length(cache_range))

    reqs, items_large, rates_large = averageRequests(T+1, 5, params.pd, params.numof_requests, 0.5)
    funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
    #SY_0 = randomInitPoint(size(probcomps.netgraph.edges,1), netparams.V*params.M, 0, probcomps.consts)
    #S_0 = (probcomps.consts.P_max / size(probcomps.netgraph.edges,1)) * ones(Float64, size(probcomps.netgraph.edges,1)) # total
    S_0 = (probcomps.consts.P_min) * ones(Float64, size(probcomps.netgraph.edges,1)) # per node
    Y_0 = zeros(Float64, netparams.V*params.M)
    SY_0 = (S_0, Y_0)

    for (i, c_sc) in enumerate(cache_range)
        c_mc = min(2*c_sc,  convert(Int64, 0.8*params.M))
        consts = makeConsts(netparams.V, params.M, findall(n -> n == 1, probcomps.V_pos[:,3]), probcomps.netgraph.edges, c_mc, c_sc, constparams.P_min, constparams.P_max)

        #weight = (i, N) -> (N != 1) * (( i - ( (N + 1) / 2) ) / (N - 1))
        #SY_0 = [ randomInitPoint(size(probcomps.netgraph.edges,1), netparams.V*params.M, weight(i, params.numof_initpoints), consts) for i in 1:params.numof_initpoints ]

        (D_opt, S_opt, Y_opt, _, _) = altMethod(SY_0, funcs, consts) # Since we're strictly increasing cache capacity, we can reuse the same initial points for ALT
        X_opt = pipageRounding(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, netparams.V, consts.cache_capacity)
        D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.Gintegral) ])
        @printf("--/ c_sc = %f  c_mc = %f /--\n",c_sc,c_mc)
        println("--ALT--")
        @printf("Delay: %.2f, Not Rounded: %.2f\n", D_0, D_opt)
        D_0, S_opt = pwrOnly(S_opt, X_opt, funcs, consts)
        @printf("One more pwr: %.2f\n",D_0)
        results[i] = D_0;

        #(D_sub, S_sub, Y_sub) = subMethodMult(SY_0, funcs, consts)
        (D_sub, S_sub, Y_sub, _, _) = subMethod(SY_0, funcs, consts)
        X_sub = pipageRounding(funcs.F, funcs.Gintegral, S_sub, Y_sub, params.M, netparams.V, consts.cache_capacity)
        D_0_sub = sum([ funcs.F[m](S_sub) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_sub) for n in 1:length(funcs.Gintegral) ])
        println("--SUB--")
        @printf("Delay: %.2f, Not Rounded: %.2f\n", D_0_sub, D_sub)
        D_0_sub, S_sub = pwrOnly(S_sub, X_sub, funcs, consts)
        @printf("One more pwr: %.2f\n",D_0_sub)
        sub_results[i] = D_0_sub;

        newprob = problem_components(probcomps.V_pos, probcomps.netgraph, reqs, funcs, consts, probcomps.SY_0)
        lru_results[i], lfu_results[i], fifo_results[i] = baselinePoSim(T, items_large, rates_large, params, netparams, newprob)
    end

    open("/home/volkan/opt-caching-power/julia/cacheresults.txt","a") do io
        println(io, "--ALT--")
        println(io, results)
        println(io, "--SUB--")
        println(io, sub_results)
        println(io, "--POLRU--")
        println(io, lru_results)
        println(io, "--POLFU--")
        println(io, lfu_results)
        println(io, "--POFIFO--")
        println(io, fifo_results)
    end

    return plot(cache_range, hcat(results,sub_results,lru_results,lfu_results,fifo_results), title="Delay with increasing c_sc", label = ["ALT" "SUB" "LRU" "LFU" "FIFO"], xlabel="c_sc", ylabel="Delay")
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
    lru_timestamps, lru_cache = lruAdvance(t, lru_timestamps, lru_cache, probcomps.reqs, paths, cache_capacity)
    t += 1
    while t <= T+1
        X_t = deepcopy(reshape(lru_cache,M*V))
        reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D_lru += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_t) for n in 1:length(funcs.Gintegral)) )
        lru_timestamps, lru_cache = lruAdvance(t, lru_timestamps, lru_cache, reqs, paths, cache_capacity)
        t += 1
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
    lfu_counts, lfu_cache = lfuAdvance(lfu_counts, lfu_cache, probcomps.reqs, paths, cache_capacity)
    t += 1
    while t <= T+1
        X_t = deepcopy(reshape(lfu_cache,M*V))
        reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D_lfu += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_t) for n in 1:length(funcs.Gintegral)) )
        lfu_counts, lfu_cache = lfuAdvance(lfu_counts, lfu_cache, reqs, paths, cache_capacity)        
        t += 1
    end
    
    println(" -- LFU --")
    @printf("Delay: %.2f\n", D_lfu/(T))
    return D_lfu/(T)
end

function lrfuSim(T::Int64, params::other_parameters = params, netparams::network_parameters = netparams, probcomps::problem_components = probcomps)
    M = params.M
    V = netparams.V
    cache_capacity = probcomps.consts.cache_capacity
    paths = probcomps.netgraph.paths
    F = probcomps.funcs.F
    Gintegral = probcomps.funcs.Gintegral

    lrfu_counts = ones(Float64, M,V)
    lrfu_timestamps = ones(Int64, M,V)
    lrfu_cache = BitArray(undef, M,V)
    lrfu_cache[:,:] .= 0
    for v in 1:V
        lrfu_cache[1:cache_capacity[v],v] .= 1
    end
    S_0 = (probcomps.consts.P_max / size(probcomps.netgraph.edges,1)) * ones(Float64, size(probcomps.netgraph.edges,1))

    X_0 = deepcopy(reshape(lrfu_cache,M*V))
    D_lrfu = ThreadsX.sum( ThreadsX.collect(F[m](S_0) for m in 1:length(F)) .* ThreadsX.collect(Gintegral[n](X_0) for n in 1:length(Gintegral)) )
    t = 2
    lrfu_scores, lrfu_timestamps, lrfu_cache = lrfuAdvance(t, lrfu_scores, lrfu_timestamps, lrfu_cache, probcomps.reqs, paths, cache_capacity)
    t += 1
    while t <= T+1
        X_t = deepcopy(reshape(lrfu_cache,M*V))
        reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D_lrfu += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_t) for n in 1:length(funcs.Gintegral)) )
        lrfu_scores, lrfu_timestamps, lrfu_cache = lrfuAdvance(t, lrfu_scores, lrfu_timestamps, lrfu_cache, reqs, paths, cache_capacity)
        t += 1
    end
    
    println(" -- LRFU --")
    @printf("Delay: %.2f\n", D_lrfu/(T))
    return D_lrfu/(T)
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
    fifo_queues, fifo_cache = fifoAdvance(fifo_queues, fifo_cache, probcomps.reqs, paths, cache_capacity)
    t += 1
    while t <= T+1
        X_t = deepcopy(reshape(fifo_cache,M*V))
        reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D_fifo += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_t) for n in 1:length(funcs.Gintegral)) )
        fifo_queues, fifo_cache = fifoAdvance(fifo_queues, fifo_cache, reqs, paths, cache_capacity)        
        t += 1
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
    rand_cache = randAdvance(rand_cache, probcomps.reqs, paths, cache_capacity)
    t += 1
    while t <= T+1
        X_t = deepcopy(reshape(rand_cache,M*V))
        reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D_rand += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_t) for n in 1:length(funcs.Gintegral)) )
        rand_cache = randAdvance(rand_cache, reqs, paths, cache_capacity)        
        t += 1
    end
    
    println(" -- RANDOM --")
    @printf("Delay: %.2f\n", D_rand/(T))
    return D_rand/(T)
end

function baselineSim(T::Int64, items_large, rates_large, params::other_parameters = params, netparams::network_parameters = netparams, probcomps::problem_components = probcomps)
    M = params.M
    V = netparams.V
    cache_capacity = probcomps.consts.cache_capacity
    paths = probcomps.netgraph.paths
    F = probcomps.funcs.F
    Gintegral = probcomps.funcs.Gintegral

    lru_timestamps = ones(Int64, M,V)
    lru_cache = BitArray(undef, M,V)
    lru_cache[:,:] .= 0

    lfu_counts = ones(Int64, M,V)
    lfu_cache = BitArray(undef, M,V)
    lfu_cache[:,:] .= 0

    #lrfu_timestamps = ones(Int64, M,V)
    #lrfu_scores = ones(Float64, M,V)
    #lrfu_cache = BitArray(undef, M,V)
    #lrfu_cache[:,:] .= 0

    fifo_queues = collect( [ Queue{Int64}() for i in 1:V] )
    fifo_cache = BitArray(undef, M,V)
    fifo_cache[:,:] .= 0

    #rand_cache = BitArray(undef, M,V)
    #rand_cache[:,:] .= 0

    for v in 1:V
        lru_cache[1:cache_capacity[v],v] .= 1
        lfu_cache[1:cache_capacity[v],v] .= 1
        #lrfu_cache[1:cache_capacity[v],v] .= 1
        for i in 1:cache_capacity[v]
            enqueue!(fifo_queues[v],i)
        end
        fifo_cache[1:cache_capacity[v],v] .= 1
        #rand_cache[1:cache_capacity[v],v] .= 1
    end

    S_0 = (probcomps.consts.P_max / size(probcomps.netgraph.edges,1)) * ones(Float64, size(probcomps.netgraph.edges,1))

    X_lru_0 = deepcopy(reshape(lru_cache,M*V))
    X_lfu_0 = deepcopy(reshape(lfu_cache,M*V))
    #X_lrfu_0 = deepcopy(reshape(lrfu_cache,M*V))
    X_fifo_0 = deepcopy(reshape(fifo_cache,M*V))
    #X_rand_0 = deepcopy(reshape(rand_cache,M*V))

    D_lru = ThreadsX.sum( ThreadsX.collect(F[m](S_0) for m in 1:length(F)) .* ThreadsX.collect(Gintegral[n](X_lru_0) for n in 1:length(Gintegral)) )
    D_lfu = ThreadsX.sum( ThreadsX.collect(F[m](S_0) for m in 1:length(F)) .* ThreadsX.collect(Gintegral[n](X_lfu_0) for n in 1:length(Gintegral)) )
    #D_lrfu = ThreadsX.sum( ThreadsX.collect(F[m](S_0) for m in 1:length(F)) .* ThreadsX.collect(Gintegral[n](X_lrfu_0) for n in 1:length(Gintegral)) )
    D_fifo = ThreadsX.sum( ThreadsX.collect(F[m](S_0) for m in 1:length(F)) .* ThreadsX.collect(Gintegral[n](X_fifo_0) for n in 1:length(Gintegral)) )
    #D_rand = ThreadsX.sum( ThreadsX.collect(F[m](S_0) for m in 1:length(F)) .* ThreadsX.collect(Gintegral[n](X_rand_0) for n in 1:length(Gintegral)) )

    t = 2
    lru_timestamps, lru_cache = lruAdvance(t, lru_timestamps, lru_cache, probcomps.reqs, paths, cache_capacity)
    lfu_counts, lfu_cache = lfuAdvance(lfu_counts, lfu_cache, probcomps.reqs, paths, cache_capacity)
    #lrfu_scores, lrfu_timestamps, lrfu_cache = lrfuAdvance(t, lrfu_scores, lrfu_timestamps, lrfu_cache, probcomps.reqs, paths, cache_capacity)
    fifo_queues, fifo_cache = fifoAdvance(fifo_queues, fifo_cache, probcomps.reqs, paths, cache_capacity)
    #rand_cache = randAdvance(rand_cache, probcomps.reqs, paths, cache_capacity)
    t += 1

    period = 20
    tc = rand(Int64(floor(period/4)):Int64(ceil(3*period/4)), params.M, params.numof_requests)
    while t <= T+1
        X_lru_t = deepcopy(reshape(lru_cache,M*V))
        X_lfu_t = deepcopy(reshape(lfu_cache,M*V))
        #X_lrfu_t = deepcopy(reshape(lrfu_cache,M*V))
        X_fifo_t = deepcopy(reshape(fifo_cache,M*V))
        #X_rand_t = deepcopy(reshape(rand_cache,M*V))
        #reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        reqs, tc = dependentRequests(t, period, tc, params.pd, params.numof_requests, 1.0)
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D_lru += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_lru_t) for n in 1:length(funcs.Gintegral)) )
        D_lfu += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_lfu_t) for n in 1:length(funcs.Gintegral)) )
        #D_lrfu += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_lrfu_t) for n in 1:length(funcs.Gintegral)) )
        D_fifo += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_fifo_t) for n in 1:length(funcs.Gintegral)) )
        #D_rand += ThreadsX.sum( ThreadsX.collect(funcs.F[m](S_0) for m in 1:length(funcs.F)) .* ThreadsX.collect(funcs.Gintegral[n](X_rand_t) for n in 1:length(funcs.Gintegral)) )
        lru_timestamps, lru_cache = lruAdvance(t, lru_timestamps, lru_cache, reqs, paths, cache_capacity)
        lfu_counts, lfu_cache = lfuAdvance(lfu_counts, lfu_cache, reqs, paths, cache_capacity)
        lrfu_scores, lrfu_timestamps, lrfu_cache = lrfuAdvance(t, lrfu_scores, lrfu_timestamps, lrfu_cache, reqs, paths, cache_capacity)
        fifo_queues, fifo_cache = fifoAdvance(fifo_queues, fifo_cache, reqs, paths, cache_capacity)
        rand_cache = randAdvance(rand_cache, reqs, paths, cache_capacity)        
        t += 1
    end

    println(" -- LRU --")
    @printf("Delay: %.2f\n", D_lru/(T))

    println(" -- LFU --")
    @printf("Delay: %.2f\n", D_lfu/(T))

    println(" -- LRFU --")
    @printf("Delay: %.2f\n", D_lrfu/(T))

    println(" -- FIFO --")
    @printf("Delay: %.2f\n", D_fifo/(T))

    println(" -- RANDOM --")
    @printf("Delay: %.2f\n", D_rand/(T))

    return D_lru/(T), D_lfu/(T), D_lrfu/(T), D_fifo/(T), D_rand/(T)
end

function baselinePoSim(T::Int64, items_large, rates_large, params::other_parameters = params, netparams::network_parameters = netparams, probcomps::problem_components = probcomps)
    M = params.M
    V = netparams.V
    cache_capacity = probcomps.consts.cache_capacity
    paths = probcomps.netgraph.paths
    F = probcomps.funcs.F
    Gintegral = probcomps.funcs.Gintegral

    lru_timestamps = ones(Int64, M,V)
    lru_cache = BitArray(undef, M,V)
    lru_cache[:,:] .= 0

    lfu_counts = ones(Int64, M,V)
    lfu_cache = BitArray(undef, M,V)
    lfu_cache[:,:] .= 0

    #lrfu_timestamps = ones(Int64, M,V)
    #lrfu_scores = ones(Float64, M,V)
    #lrfu_cache = BitArray(undef, M,V)
    #lrfu_cache[:,:] .= 0

    fifo_queues = collect( [ Queue{Int64}() for i in 1:V] )
    fifo_cache = BitArray(undef, M,V)
    fifo_cache[:,:] .= 0

    #rand_cache = BitArray(undef, M,V)
    #rand_cache[:,:] .= 0

    for v in 1:V
        lru_cache[1:cache_capacity[v],v] .= 1
        lfu_cache[1:cache_capacity[v],v] .= 1
        #lrfu_cache[1:cache_capacity[v],v] .= 1
        for i in 1:cache_capacity[v]
            enqueue!(fifo_queues[v],i)
        end
        fifo_cache[1:cache_capacity[v],v] .= 1
        #rand_cache[1:cache_capacity[v],v] .= 1
    end

    #S_0 = (probcomps.consts.P_max / size(probcomps.netgraph.edges,1)) * ones(Float64, size(probcomps.netgraph.edges,1)) # total
    S_0 = (probcomps.consts.P_min) * ones(Float64, size(probcomps.netgraph.edges,1)) # per node

    X_lru_0 = deepcopy(reshape(lru_cache,M*V))
    X_lfu_0 = deepcopy(reshape(lfu_cache,M*V))
    #X_lrfu_0 = deepcopy(reshape(lrfu_cache,M*V))
    X_fifo_0 = deepcopy(reshape(fifo_cache,M*V))
    #X_rand_0 = deepcopy(reshape(rand_cache,M*V))

    D_lru, S_lru_t = pwrOnly(S_0, X_lru_0, probcomps.funcs, probcomps.consts)
    D_lfu, S_lfu_t = pwrOnly(S_0, X_lfu_0, probcomps.funcs, probcomps.consts)
    #D_lrfu, S_lrfu_t = pwrOnly(S_0, X_lrfu_0, probcomps.funcs, probcomps.consts)
    D_fifo, S_fifo_t = pwrOnly(S_0, X_fifo_0, probcomps.funcs, probcomps.consts)
    #D_rand, S_rand_t = pwrOnly(S_0, X_rand_0, probcomps.funcs, probcomps.consts)

    t = 2
    lru_timestamps, lru_cache = lruAdvance(t, lru_timestamps, lru_cache, probcomps.reqs, paths, cache_capacity)
    lfu_counts, lfu_cache = lfuAdvance(lfu_counts, lfu_cache, probcomps.reqs, paths, cache_capacity)
    #lrfu_scores, lrfu_timestamps, lrfu_cache = lrfuAdvance(t, lrfu_scores, lrfu_timestamps, lrfu_cache, probcomps.reqs, paths, cache_capacity)
    fifo_queues, fifo_cache = fifoAdvance(fifo_queues, fifo_cache, probcomps.reqs, paths, cache_capacity)
    #rand_cache = randAdvance(rand_cache, probcomps.reqs, paths, cache_capacity)
    t += 1

    period = 5
    #tc = rand(Int64(floor(period/4)):Int64(ceil(3*period/4)), params.M, params.numof_requests)
    while t <= T+1
        X_lru_t = deepcopy(reshape(lru_cache,M*V))
        X_lfu_t = deepcopy(reshape(lfu_cache,M*V))
        #X_lrfu_t = deepcopy(reshape(lrfu_cache,M*V))
        X_fifo_t = deepcopy(reshape(fifo_cache,M*V))
        #X_rand_t = deepcopy(reshape(rand_cache,M*V))
        #reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        #reqs, tc = dependentRequests(t, period, tc, params.pd, params.numof_requests, 0.5)
        reqs = requests(items_large[t,:], rates_large[t,:])
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        #D, S_lru_t = pwrOnly(S_lru_t, X_lru_t, funcs, probcomps.consts)
        lru_task = @spawn pwrOnly(S_lru_t, X_lru_t, funcs, probcomps.consts)
        #D_lru += D
        #D, S_lfu_t = pwrOnly(S_lfu_t, X_lfu_t, funcs, probcomps.consts)
        lfu_task = @spawn pwrOnly(S_lfu_t, X_lfu_t, funcs, probcomps.consts)
        #D_lfu += D
        #D, S_lrfu_t = pwrOnly(S_lrfu_t, X_lrfu_t, funcs, probcomps.consts)
        #D_lrfu += D
        #D, S_fifo_t = pwrOnly(S_fifo_t, X_fifo_t, funcs, probcomps.consts)
        fifo_task = @spawn pwrOnly(S_fifo_t, X_fifo_t, funcs, probcomps.consts)
        #D_fifo += D
        #D, S_rand_t = pwrOnly(S_rand_t, X_rand_t, funcs, probcomps.consts)
        #rand_task = @spawn pwrOnly(S_rand_t, X_rand_t, funcs, probcomps.consts)
        #D_rand += D
        lru_timestamps, lru_cache = lruAdvance(t, lru_timestamps, lru_cache, reqs, paths, cache_capacity)
        lfu_counts, lfu_cache = lfuAdvance(lfu_counts, lfu_cache, reqs, paths, cache_capacity)
        #lrfu_scores, lrfu_timestamps, lrfu_cache = lrfuAdvance(t, lrfu_scores, lrfu_timestamps, lrfu_cache, reqs, paths, cache_capacity)
        fifo_queues, fifo_cache = fifoAdvance(fifo_queues, fifo_cache, reqs, paths, cache_capacity)
        #rand_cache = randAdvance(rand_cache, reqs, paths, cache_capacity)
        D, S_lru_t = fetch(lru_task)
        D_lru += D
        D, S_lfu_t = fetch(lfu_task)
        D_lfu += D
        D, S_fifo_t = fetch(fifo_task)
        D_fifo += D
        #D, S_rand_t = fetch(rand_task)
        #D_rand += D
        t += 1
    end

    println(" -- LRU --")
    @printf("Delay: %.2f\n", D_lru/(T))

    println(" -- LFU --")
    @printf("Delay: %.2f\n", D_lfu/(T))

    #println(" -- LRFU --")
    #@printf("Delay: %.2f\n", D_lrfu/(T))

    println(" -- FIFO --")
    @printf("Delay: %.2f\n", D_fifo/(T))

    #println(" -- RANDOM --")
    #@printf("Delay: %.2f\n", D_rand/(T))

    #return D_lru/(T), D_lfu/(T), D_lrfu/(T), D_fifo/(T), D_rand/(T)
    return D_lru/(T), D_lfu/(T), D_fifo/(T)
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
    lru_timestamps, lru_cache = lruAdvance(t, lru_timestamps, lru_cache, probcomps.reqs, paths, cache_capacity)
    t += 1

    period = 5
    tc = rand(Int64(floor(period/4)):Int64(ceil(3*period/4)), params.M, params.numof_requests)
    while t <= T+1
        X_t = deepcopy(reshape(lru_cache,M*V))
        #reqs = randomRequests(params.pd, params.numof_requests, 1.0)
        reqs, tc = dependentRequests(t, period, tc, params.pd, params.numof_requests, 1.0)
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D, S_t = pwrOnly(S_t, X_t, funcs, probcomps.consts)
        D_polru += D
        lru_timestamps, lru_cache = lruAdvance(t, lru_timestamps, lru_cache, reqs, paths, cache_capacity)
        t += 1
    end
    
    println(" -- POLRU --")
    @printf("Delay: %.2f\n", D_polru/(T))
end

function polfuSim(T::Int64, items_large, rates_large, params::other_parameters = params, netparams::network_parameters = netparams, probcomps::problem_components = probcomps)
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
    X_0 = deepcopy(reshape(lru_cache,M*V))
    D_polfu, S_t = pwrOnly(S_0, X_0, probcomps.funcs, probcomps.consts)

    t = 2
    lfu_counts, lfu_cache = lfuAdvance(lfu_counts, lru_cache, probcomps.reqs, paths, cache_capacity)
    t += 1

    while t <= T+1
        X_t = deepcopy(reshape(lru_cache,M*V))
        reqs = requests(items_large[t,:], rates_large[t,:])
        funcs = funcSetup(probcomps.netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        D, S_t = pwrOnly(S_t, X_t, funcs, probcomps.consts)
        D_polfu += D
        lfu_counts, lfu_cache = lfuAdvance(lfu_counts, lfu_cache, reqs, paths, cache_capacity)
        t += 1
    end
    
    println(" -- POLFU --")
    @printf("Delay: %.2f\n", D_polfu/(T))
end

function incScCombSim(inc_count::Int64, T::Int64, params::other_parameters = params, netparams::network_parameters = netparams, constparams::constraint_parameters = constparams, probcomps::problem_components = probcomps)
    open("/home/volkan/opt-caching-power/julia/longresults.txt","a") do io
        println(io, "--------")
        println(io, "--params--")
        println(io, params)
        println(io, "--netparams--")
        println(io, netparams)
        println(io, "--constparams--")
        println(io, constparams)
    end
    results = zeros(Float64, inc_count+1)
    rcf_results = zeros(Float64, inc_count+1)
    sub_results = zeros(Float64, inc_count+1)
    lru_results = zeros(Float64, inc_count+1)
    lfu_results = zeros(Float64, inc_count+1)
    #lrfu_results = zeros(Float64, inc_count+1)
    fifo_results = zeros(Float64, inc_count+1)
    #rand_results = zeros(Float64, inc_count+1)

    V_pos = probcomps.V_pos
    netgraph = probcomps.netgraph
    reqs, items_large, rates_large = averageRequests(T+1, 5, params.pd, params.numof_requests, 0.5)
    funcs = funcSetup(netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
    #reqs = probcomps.reqs
    #funcs = probcomps.funcs
    consts = probcomps.consts
    #SY_0 = probcomps.SY_0

    S_0 = (probcomps.consts.P_max / size(probcomps.netgraph.edges,1)) * ones(Float64, size(probcomps.netgraph.edges,1))
    Y_0 = zeros(Float64, netparams.V*params.M)
    SY_0 = (S_0, Y_0)

    (D_opt, S_opt, Y_opt) = altMethod(SY_0, funcs, consts)
    X_opt = pipageRounding(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, netparams.V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("--/ SC = %d /--\n",netparams.SC)
    println("--ALT--")
    @printf("Delay: %.2f, Not Rounded: %.2f\n", D_0, D_opt)
    D_0, S_opt = pwrOnly(S_opt, X_opt, funcs, consts)
    @printf("One more pwr: %.2f\n",D_0)
    rcf_results[1] = D_opt;
    results[1] = D_0;

    #(D_sub, S_sub, Y_sub) = subMethodMult(SY_0, funcs, consts)
    (D_sub, S_sub, Y_sub) = subMethod(SY_0, funcs, consts)
    X_sub = pipageRounding(funcs.F, funcs.Gintegral, S_sub, Y_sub, params.M, netparams.V, consts.cache_capacity)
    D_0_sub = sum([ funcs.F[m](S_sub) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_sub) for n in 1:length(funcs.Gintegral) ])
    println("--SUB--")
    @printf("Delay: %.2f, Not Rounded: %.2f\n", D_0_sub, D_sub)
    D_0_sub, S_sub = pwrOnly(S_sub, X_sub, funcs, consts)
    @printf("One more pwr: %.2f\n",D_0_sub)
    sub_results[1] = D_0_sub;

    #lru_results[1], lfu_results[1], lrfu_results[1], fifo_results[1], rand_results[1] = baselineSim(T)
    #lru_results[1], lfu_results[1], lrfu_results[1], fifo_results[1], rand_results[1] = baselinePoSim(T)
    newprob = problem_components(V_pos, netgraph, reqs, funcs, consts, probcomps.SY_0)
    lru_results[1], lfu_results[1], fifo_results[1] = baselinePoSim(T, items_large, rates_large, params, netparams, newprob)

    for i in 2:inc_count+1
        netparams.V += 1
        netparams.SC += 1
        #V_pos = addSC(V_pos, netparams.R_cell)
        #V_pos = addVoronoiSC(V_pos, netparams.R_cell)
        V_pos = voronoiDist(netparams.V, netparams.SC, netparams.R_cell)
        netgraph = makeGraph(V_pos, netparams.pathloss_exponent, netparams.C_bh_mc, netparams.C_bh_sc)
        funcs = funcSetup(netgraph, reqs, netparams.V, params.M, netparams.noise, params.D_bh_mc, params.D_bh_sc)
        consts = makeConsts(netparams.V, params.M, constparams.c_mc, constparams.c_sc, findall(i -> i == 1, V_pos[:,3]), constparams.P_min, constparams.P_max)
        #SY_0 = randomInitPoint(size(netgraph.edges,1), netparams.V*params.M, 0, consts)

        #weight = (i, N) -> (N != 1) * (( i - ( (N + 1) / 2) ) / (N - 1))
        #SY_0 = [ randomInitPoint(size(netgraph.edges,1), netparams.V*params.M, weight(i, params.numof_initpoints), consts) for i in 1:params.numof_initpoints ]
        S_0 = (probcomps.consts.P_max / size(netgraph.edges,1)) * ones(Float64, size(netgraph.edges,1))
        Y_0 = zeros(Float64, netparams.V*params.M)
        SY_0 = (S_0, Y_0)

        (D_opt, S_opt, Y_opt) = altMethod(SY_0, funcs, consts)
        X_opt = pipageRounding(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, netparams.V, consts.cache_capacity)
        D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
        @printf("--/ SC = %d /--\n",netparams.SC)
        println("--ALT--")
        @printf("Delay: %.2f, Not Rounded: %.2f\n", D_0, D_opt)
        D_0, S_opt = pwrOnly(S_opt, X_opt, funcs, consts)
        @printf("One more pwr: %.2f\n",D_0)
        rcf_results[i] = D_opt;
        results[i] = D_0;

        #(D_sub, S_sub, Y_sub) = subMethodMult(SY_0, funcs, consts)
        (D_sub, S_sub, Y_sub) = subMethod(SY_0, funcs, consts)
        X_sub = pipageRounding(funcs.F, funcs.Gintegral, S_sub, Y_sub, params.M, netparams.V, consts.cache_capacity)
        D_0_sub = sum([ funcs.F[m](S_sub) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_sub) for n in 1:length(funcs.Gintegral) ])
        println("--SUB--")
        @printf("Delay: %.2f, Not Rounded: %.2f\n", D_0_sub, D_sub)
        D_0_sub, S_sub = pwrOnly(S_sub, X_sub, funcs, consts)
        @printf("One more pwr: %.2f\n",D_0_sub)
        sub_results[i] = D_0_sub;

        newprob = problem_components(V_pos, netgraph, reqs, funcs, consts, probcomps.SY_0)
        #lru_results[i], lfu_results[i], lrfu_results[i], fifo_results[i], rand_results[i] = baselineSim(T, params, netparams, newprob)
        #lru_results[i], lfu_results[i], lrfu_results[i], fifo_results[i], rand_results[i] = baselinePoSim(T, params, netparams, newprob)
        lru_results[i], lfu_results[i], fifo_results[i] = baselinePoSim(T, items_large, rates_large, params, netparams, newprob)
    end

    open("/home/volkan/opt-caching-power/julia/longresults.txt","a") do io
        println(io, "--ALT--")
        println(io, results)
        println(io, "--ALT (RCF)--")
        println(io, rcf_results)
        println(io, "--SUB--")
        println(io, sub_results)
        println(io, "--POLRU--")
        println(io, lru_results)
        println(io, "--POLFU--")
        println(io, lfu_results)
        println(io, "--POFIFO--")
        println(io, fifo_results)
    end
    #return plot((netparams.SC - inc_count):netparams.SC, hcat(results,lru_results,lfu_results,lrfu_results,fifo_results,rand_results), title="Delay with increasing # of SCs", label = ["ALT" "LRU" "LFU" "LRFU" "FIFO" "RANDOM"], xlabel="# of SCs", ylabel="Delay")
    return plot((netparams.SC - inc_count):netparams.SC, hcat(results,rcf_results,sub_results,lru_results,lfu_results,fifo_results), title="Delay with increasing # of SCs", label = ["ALT" "ALT-RCF" "SUB" "LRU" "LFU" "FIFO"], xlabel="# of SCs", ylabel="Delay")
end