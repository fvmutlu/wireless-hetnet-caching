include("helper_functions.jl")

P_min = 1
P_max = 10

M = 2
V = 1

dim_S = 3
dim_Y = 2

cache_capacity = 1
C = [1 1]
weight = 0.5

(S_0, Y_0) = randomInitPoint(dim_S, dim_Y, P_min, P_max, C, cache_capacity, weight)
println(S_0)
println(Y_0)

function F(S)
    lambda1 = 1.2
    lambda2 = 1.5
    sinr1 = S[1]/(1+S[2]+S[3]/16)
    sinr2 = S[2]/(1+S[1]+S[3]/16)
    sinr3 = S[3]
    f1 = 1/log(1+sinr1)
    f2 = 1/log(1+sinr2)
    f3 = 1/log(1+sinr3)
    return [lambda1*f1 lambda1*f3 lambda2*f2 lambda2*f3]
end

function Gintegral(Y)
    g1 = 1 - Y[1]
    g2 = 1 - Y[2]
    return [1 g1 1 g2]
end

pipageRound(F, Gintegral, Y_0, S_0, M, V, cache_capacity)