include("methods.jl")

using Printf

mutable struct other_parameters
    M::Int64 # Total number of items in the catalog
    pd::Array{Float64,1} # Probability distribution for requests
    numof_requests::Int64 # Total number of requests across all users (NAMED THIS WAY FOR FUTURE CONSIDERATION, FOR CURRENT CASE THIS IS ALWAYS EQUAL TO NUMBER OF USERS) 
    D_bh_mc::Float64 # MC-BH wireline link delay (LEFT HERE FOR POSSIBLE CUSTOM ADJUSTMENT, CURRENTLY CALCULATED VIA HEURISTICS USING POWER CONSTRAINTS AND NETWORK PARAMETERS)
    D_bh_sc::Float64 # SC-BH wireline link delay, same for all SCs (LEFT HERE FOR POSSIBLE CUSTOM ADJUSTMENT, CURRENTLY CALCULATED VIA HEURISTICS USING POWER CONSTRAINTS AND NETWORK PARAMETERS)
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

function readConfig(input = "/home/volkan/opt-caching-power/julia/config.txt")
    cfg = split.(readlines(input), " # ")
    (V, SC, M, c_mc, c_sc, pd_type) = parse.(Int64, [ cfg[i][1] for i in 1:6 ])
    (P_min, P_max, pathloss_exponent, noise, R_cell, pd_param) = parse.(Float64, [ cfg[i][1] for i in 7:12 ])
    
    U = V - SC - 1
    C_bh_mc = (R_cell/2)^pathloss_exponent # Cost at the wireline edge between backhaul to macro cell (calculated completely heuristically, there could be smarter way of determining this)
    C_bh_sc = 1.5 * C_bh_mc # Cost at the wireline edge between backhaul and any of the small cells (again, heuristic)
    D_bh_mc = 1 / log( 1 + ((1/C_bh_mc)*P_max)/noise ) # Delay of data retrieval from backhaul to macro cell (again, computed heuristically using the wireline edge costs)
    D_bh_sc = 1 / log( 1 + ((1/C_bh_sc)*P_max*0.8)/noise ) # Delay of data retrieval from backhaul to any of the small cells (again, computed heuristically using the wireline edge costs)
    
    if pd_type == 0
        pd = (1 ./ (1:M).^ pd_param) / sum(1 ./ (1:M).^ pd_param)
    else # have different distributions here possibly?
        pd = (1 ./ (1:M).^ pd_param) / sum(1 ./ (1:M).^ pd_param)
    end
    
    params = other_parameters(M, pd, U, D_bh_mc, D_bh_sc) # numof_requests = U FOR CURRENT CASE
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

    return V_pos, netgraph, reqs, funcs, consts
end

## Run optimization over different initial points

function runSim(numof_initpoints::Int64, numof_edges::Int64 = size(netgraph.edges,1), V::Int64 = netparams.V, M::Int64 = params.M, consts::constraints = consts, funcs = funcs)
    weight = (i, N) -> (N != 1) * (( i - ( (N + 1) / 2) ) / (N - 1))
    SY_0 = [ randomInitPoint(numof_edges, V*M, weight(i, numof_initpoints), consts) for i in 1:numof_initpoints ]

    println(" -- SUB -- ")
    (D_opt, S_opt, Y_opt) = @time subMethod(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, M, V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0)

    println(" -- SUB MULT -- ")
    (D_opt, S_opt, Y_opt) = @time subMethodMult(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, M, V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0)

    println(" -- ALT --")
    (D_opt, S_opt, Y_opt) = @time altMethod(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, M, V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0)

    println(" -- ALT MULT -- ")
    (D_opt, S_opt, Y_opt) = @time altMethodMult(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, M, V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0)
end