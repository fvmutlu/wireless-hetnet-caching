V = 10
SC = 3
U = V - SC - 1
M = 5
D_bh_mc = 1.5
D_bh_sc = 2.5
c_mc = 3
c_sc = 2
P_min = 1
P_max = 15

params = (noise = 1.0, numof_requests = U, V = V, M = M, D_bh_mc = D_bh_mc, D_bh_sc = D_bh_sc)
netgraph = cellGraph(10, 3, 5.0, 4.0, 200.0, 100.0)

numof_edges = size(netgraph.edges, 1)
numof_hops = sum(length.(netgraph.paths)) - length(netgraph.paths)

numof_initpoints = 10

S_0 = [ 10*rand(Float64, numof_edges) for i in 1:numof_initpoints ]
Y_0 = [ rand(Float64, 50) for i in 1:numof_initpoints ]

cc = zeros(Int64, V,1)
cc[1] = c_mc
cc[netgraph.sc_nodes, 1] .= c_sc

Cmat = zeros(Int64, V,M*V) # For cache capacity constraint C*Y <= cache_capacity
for n in 1:V
    Cmat[ n, (n-1)*M+1 : n*M ] .= 1 # In matrix C, mark entries corresponding to node n's items as 1
end

consts = (P_min = P_min, P_max = P_max, cache_capacity = cc, C = Cmat)

funcs = funcSetup(netgraph, params)

D_opt = subMethod(S_0, Y_0, funcs, consts)
    
    