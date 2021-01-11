using SymEngine

function grad(F, x)
    rows = size(x, 1)
    cols = size(F, 1)
    gradF = [symbols("gradF_$i") for i in 1:rows, j in 1:cols]
    for i in 1:rows, j in 1:cols
        gradF[i, j] = diff(F[j], x[i])
    end
    return gradF
end

#= function vecSubs(F,x,y) # May not be obsolete with matSubs
    rows = size(x,1)
    if ! (rows == size(y,1))
        error("Matrix substitution error: dimensions do not match.")
        return 0
    end
    result = F
    for i in 1:rows
        result = subs.(result,x[i]=>y[i])
    end
    return result
end =#

function matSubs(F, x, y)
    rows = size(x, 1)
    cols = size(x, 2)
    if ! (rows == size(y, 1) && cols == size(y, 2))
        error("Matrix substitution error: dimensions do not match.")
        return 0
    end
    result = F
    for i in 1:rows, j in 1:cols
        result = subs.(result, x[i, j]=>y[i, j])
    end
    return result
end

function isbinary(x)
    if (x==0 || x==1)
        return true
    else
        return false
    end
end