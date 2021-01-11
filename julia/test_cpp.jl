using SymEngine

include("basic_functions.jl")

#= x = symbols(:x)
function loopSin(n)
    for i in 0:n
        a = sin(pi/(x^log(i+1)))
        a = a(x=>1.02)
    end
end

@time loopSin(10)
@time loopSin(100)
@time loopSin(1000) =#

x = [symbols("x_$i") for i in 1:5]
F = [x[1]*x[2], x[1]^2, x[4]^2 - x[3] + 2*x[1], x[3]^3, x[5]-x[2]^2+5*x[3], x[4]^4, x[1]+x[3]+x[5]]

#println(diff(F[1],x[1]))
#gradF = @time grad(F,x)
a = [1; 2; 3; 4; 5]
b = ones(Int64,1,5)
b*x
#subs.(F,x[1]=>a[1])
#@time matSubs(gradF,x,a)

    
    