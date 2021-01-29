import Base.Threads.@threads

function foo(x, N)
    result = Array{Float64}(undef,N)
    for i in 1:N
        result[i] = x^(log(i))
    end
    return result
end

function fooMult(x, N)
    result = Array{Float64}(undef,N)
    Threads.@threads for i in 1:N
        result[i] = x^(log(i))
    end
    return result
end