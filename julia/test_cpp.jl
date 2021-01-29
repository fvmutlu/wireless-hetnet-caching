include("methods.jl")
import Base.Threads.@spawn
using Printf

## Constraint parameters
c_mc = 4 # Cache capacity of macro cell in number of items
c_sc = 2 # Cache capacity of small cell in number of items
P_min = 1 # Minimum allowed power on each transmission edge
P_max = 200 # Maximum allowed power for the entire network / each transmission edge (dependent on projOpt function in helper_functions.jl)

## Topology and problem parameters
V = 20 # Total number of wireless nodes
SC = 4 # Number of small cells
U = V - SC - 1 # Number of users
R_cell = 4.0 # Radius of cell area
pathloss_exponent = 4.0 # Rate at which the received signal strength decreases with distance
noise = 1.0 # White noise within cell area
C_mc = (R_cell/2)^pathloss_exponent # Cost at the wireline edge between backhaul to macro cell (calculated completely heuristically, there could be smarter way of determining this)
C_sc = 1.5 * C_mc # Cost at the wireline edge between backhaul and any of the small cells (again, heuristic)
D_bh_mc = 1 / log( 1 + ((1/C_mc)*P_max)/noise ) # Delay of data retrieval from backhaul to macro cell (again, computed heuristically using the wireline edge costs)
D_bh_sc = 1 / log( 1 + ((1/C_sc)*P_max*0.8)/noise ) # Delay of data retrieval from backhaul to any of the small cells (again, computed heuristically using the wireline edge costs)
M = 10 # Size of item catalog set
zipf_exp = 0.1
pd = (1 ./ (1:M).^ zipf_exp) / sum(1 ./ (1:M).^ zipf_exp)

## Graph, functions and constraints setup
netgraph = cellGraph(V, SC, R_cell, pathloss_exponent, C_mc, C_sc)
params = (noise = noise, numof_requests = U, V = V, M = M, pd = pd, D_bh_mc = D_bh_mc, D_bh_sc = D_bh_sc)
funcs = funcSetup(netgraph, params)

cc = zeros(Int64, V,1) # For cache capacity constraint C*Y <= cache_capacity
cc[1] = c_mc
cc[netgraph.sc_nodes, 1] .= c_sc
Cmat = zeros(Int64, V,M*V) # For cache capacity constraint C*Y <= cache_capacity
for n in 1:V
    Cmat[ n, (n-1)*M+1 : n*M ] .= 1 # In matrix C, mark entries corresponding to node n's items as 1
end
consts = (P_min = P_min, P_max = P_max, cache_capacity = cc, C = Cmat)

## Run optimization over different initial points

function runSim(numof_edges, numof_initpoints, funcs, consts, params)
    weight = (i, N) -> ( i - ( (N + 1) / 2) ) / (N - 1)
    SY_0 = [ randomInitPoint(numof_edges, M*V, weight(i, numof_initpoints), consts) for i in 1:numof_initpoints ]
    println(" -- SUB -- ")
    (D_opt, S_opt, Y_opt) = @time subMethod(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, params.V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0)
    println(" -- SUBMULT -- ")
    (D_opt, S_opt, Y_opt) = @time subMethodMult(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, params.V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0)
    println(" -- ALT --")
    (D_opt, S_opt, Y_opt) = @time altMethod(SY_0, funcs, consts)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, params.V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0)
end

function runSimMult(numof_edges, numof_initpoints, funcs, consts, params)
    weight = (i, N) -> ( i - ( (N + 1) / 2) ) / (N - 1)
    SY_0 = [ randomInitPoint(numof_edges, M*V, weight(i, numof_initpoints), consts) for i in 1:numof_initpoints ]
    println(" -- Simulation begins -- ")
    t1 = @spawn subMethod(SY_0, funcs, consts)
    t2 = @spawn altMethod(SY_0, funcs, consts)
    println(" -- SUB -- ")
    (D_opt, S_opt, Y_opt) = fetch(t1)
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, params.V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0)
    println(" -- ALT -- ")
    (D_opt, S_opt, Y_opt) = fetch(t2)    
    X_opt = pipageRound(funcs.F, funcs.Gintegral, S_opt, Y_opt, params.M, params.V, consts.cache_capacity)
    D_0 = sum([ funcs.F[m](S_opt) for m in 1:length(funcs.F) ] .* [ funcs.Gintegral[n](X_opt) for n in 1:length(funcs.G) ])
    @printf("Relaxed delay: %.2f || Rounded delay: %.2f\n", D_opt, D_0)
end