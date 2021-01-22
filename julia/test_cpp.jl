include("symbolic_setup.jl")

using ForwardDiff

params = (noise = 1.0, numof_requests = 6, V = 10, M = 5, D_bh_mc = 1.5, D_bh_sc = 2.5)
netgraph = cellGraph(10, 3, 5.0, 4.0, 200.0, 100.0)

numof_edges = size(netgraph.edges, 1)
numof_hops = sum(length.(netgraph.paths)) - length(netgraph.paths)

S = 10*rand(Float64, numof_edges)
Y = rand(Float64, 50)

SINR, F, Gprime, grad_S_F, subgrad_Y_G = symSetup(netgraph, params)

println(typeof(grad_S_F))
println(typeof(subgrad_Y_G))

# hcat([ grad_S_F[i](S) for i in 1:numof_hops ]...)
    
    