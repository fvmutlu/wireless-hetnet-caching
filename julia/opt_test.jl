include("helper_functions.jl")

P_min = 1
P_max = 10

dim_S = 2
dim_Y = 4

cache_capacity = [1; 2]
C = [1 1 0 0; 0 0 1 1]
weight = 0.5

(S_0, Y_0) = randomInitPoint(dim_S, dim_Y, P_min, P_max, C, cache_capacity, weight)
(S_proj_t, Y_proj_t) = projOpt(S_0, P_min, P_max, Y_0, C, cache_capacity)