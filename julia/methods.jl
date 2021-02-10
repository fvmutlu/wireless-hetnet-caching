include("function_setup.jl")
import Base.Threads.@threads
using LinearAlgebra

const epsilon = 1e-2
const epsilon_S = 0.5*1e-2
const epsilon_Y = 0.5*1e-3

function subMethod(SY_0, funcs, consts)
    numof_initpoints = length(SY_0)
    F = funcs.F
    grad_S_F = funcs.grad_S_F
    G = funcs.G
    subgrad_Y_G = funcs.subgrad_Y_G

    D_best_sub = Inf
    S_best_sub = SY_0[1][1]
    Y_best_sub = SY_0[1][1]
    for (S_0, Y_0) in SY_0
        D_0 = sum([ F[m](S_0) for m in 1:length(F) ] .* [ G[n](Y_0) for n in 1:length(G) ])
        S_t = S_0
        Y_t = Y_0
        S_t_prev = S_t
        Y_t_prev = Y_t
        D_t = D_0

        D_hat_t = Inf
        delta = D_t/2
        div_ctr = 0
        dim_val = 2
        slow_ctr = 0
        t = 0
        while t == 0 || ( (abs(D_hat_t - D_t) >= epsilon) && (norm(Y_t_prev - Y_t) >= epsilon_Y) && (norm(S_t_prev - S_t) >= epsilon_S) )
            D_hat_t = min(D_t, D_hat_t) # If current objective is the minimum so far, replace D_hat
            d_S_t = hcat([ grad_S_F[m](S_t) for m in 1:length(grad_S_F) ]...) * [ G[n](Y_t) for n in 1:length(G) ] # Gradient of D w.r.t S evaluated at (Yₜ,Sₜ)
            d_Y_t = hcat([ subgrad_Y_G[m](Y_t) for m in 1:length(subgrad_Y_G) ]...) * ([ F[n](S_t) for n in 1:length(F) ]) # Subgradient of D w.r.t Y evaluated at (Y_t,S_t)
            step_size_S = (D_t - D_hat_t + delta) / (norm(d_S_t)^2) # Polyak step size calculation
            step_size_Y = (D_t - D_hat_t + delta) / (norm(d_Y_t)^2) # Polyak step size calculation
            S_step_t = S_t - step_size_S*d_S_t # Take step for S
            Y_step_t = Y_t - step_size_Y*d_Y_t # Take step for Y
            S_proj_t, Y_proj_t = projOpt(S_step_t, Y_step_t, consts); # Projection
            S_t_prev = S_t # We need to save S^t for the while condition
            Y_t_prev = Y_t # We need to save Y^t for the while condition
            S_t = S_proj_t # S^{t+1} = \bar{S}^t
            Y_t = Y_proj_t # Y^{t+1} = \bar{Y}^t
            D_t = sum([ F[m](S_t) for m in 1:length(F) ] .* [ G[n](Y_t) for n in 1:length(G) ]) # Calculate the objective for iteration t
            if D_t > D_hat_t # Decrease delta when zigzagging occurs
                div_ctr += 1
                if div_ctr==5
                    delta = delta / sqrt(dim_val)
                    dim_val += 1
                    div_ctr = 0;
                    slow_ctr = 0;
                end
            elseif D_hat_t - D_t <= 10*epsilon
                slow_ctr += 1
                if slow_ctr==5  # Increase delta when convergence is slow
                    delta = delta * sqrt(dim_val)
                    slow_ctr = 0
                    div_ctr = 0
                end
            end
            t += 1
        end
        if D_t < D_best_sub # Set new best point out of all initial points
            D_best_sub = D_t
            S_best_sub = S_t
            Y_best_sub = Y_t
        end
    end
    return (D_best_sub, S_best_sub, Y_best_sub)
end

function subMethodMult(SY_0, funcs, consts)
    numof_initpoints = length(SY_0)
    F = funcs.F
    grad_S_F = funcs.grad_S_F
    G = funcs.G
    subgrad_Y_G = funcs.subgrad_Y_G

    D_sub_arr = Array{Float64}(undef, numof_initpoints)
    S_sub_arr = Array{Array{Float64,1}}(undef, numof_initpoints)
    Y_sub_arr = Array{Array{Float64,1}}(undef, numof_initpoints)
    Threads.@threads for (i, (S_0, Y_0)) in collect(enumerate(SY_0))
        D_0 = sum([ F[m](S_0) for m in 1:length(F) ] .* [ G[n](Y_0) for n in 1:length(G) ])
        S_t = S_0
        Y_t = Y_0
        S_t_prev = S_t
        Y_t_prev = Y_t
        D_t = D_0

        D_hat_t = Inf
        delta = D_t/2
        div_ctr = 0
        dim_val = 2
        slow_ctr = 0
        t = 0
        while t == 0 || ( (abs(D_hat_t - D_t) >= epsilon) && (norm(Y_t_prev - Y_t) >= epsilon_Y) && (norm(S_t_prev - S_t) >= epsilon_S) )
            D_hat_t = min(D_t, D_hat_t) # If current objective is the minimum so far, replace D_hat
            d_S_t = hcat([ grad_S_F[m](S_t) for m in 1:length(grad_S_F) ]...) * [ G[n](Y_t) for n in 1:length(G) ] # Gradient of D w.r.t S evaluated at (Yₜ,Sₜ)
            d_Y_t = hcat([ subgrad_Y_G[m](Y_t) for m in 1:length(subgrad_Y_G) ]...) * ([ F[n](S_t) for n in 1:length(F) ]) # Subgradient of D w.r.t Y evaluated at (Y_t,S_t)
            step_size_S = (D_t - D_hat_t + delta) / (norm(d_S_t)^2) # Polyak step size calculation
            step_size_Y = (D_t - D_hat_t + delta) / (norm(d_Y_t)^2) # Polyak step size calculation
            S_step_t = S_t - step_size_S*d_S_t # Take step for S
            Y_step_t = Y_t - step_size_Y*d_Y_t # Take step for Y
            S_proj_t, Y_proj_t = projOpt(S_step_t, Y_step_t, consts); # Projection
            S_t_prev = S_t # We need to save S^t for the while condition
            Y_t_prev = Y_t # We need to save Y^t for the while condition
            S_t = S_proj_t # S^{t+1} = \bar{S}^t
            Y_t = Y_proj_t # Y^{t+1} = \bar{Y}^t
            D_t = sum([ F[m](S_t) for m in 1:length(F) ] .* [ G[n](Y_t) for n in 1:length(G) ]) # Calculate the objective for iteration t
            if D_t > D_hat_t # Decrease delta when zigzagging occurs
                div_ctr += 1
                if div_ctr==5
                    delta = delta / sqrt(dim_val)
                    dim_val += 1
                    div_ctr = 0;
                    slow_ctr = 0;
                end
            elseif D_hat_t - D_t <= 10*epsilon
                slow_ctr += 1
                if slow_ctr==5  # Increase delta when convergence is slow
                    delta = delta * sqrt(dim_val)
                    slow_ctr = 0
                    div_ctr = 0
                end
            end
            t += 1
        end
        D_sub_arr[i] = D_t
        S_sub_arr[i] = S_t
        Y_sub_arr[i] = Y_t
    end
    D_best_sub = minimum(D_sub_arr)
    S_best_sub = S_sub_arr[argmin(D_sub_arr)]
    Y_best_sub = Y_sub_arr[argmin(D_sub_arr)]
    return (D_best_sub, S_best_sub, Y_best_sub)
end

function altMethod(SY_0, funcs, consts) # TODO: Remove outermost for loop (don't need multiple initial points)
    numof_initpoints = length(SY_0)
    F = funcs.F
    G = funcs.G
    @assert length(F) == length(G) "F and G functions don't have the same length"
    grad_S_F = funcs.grad_S_F    
    subgrad_Y_G = funcs.subgrad_Y_G

    D_best_alt = Inf
    S_best_alt = SY_0[1][1]
    Y_best_alt = SY_0[1][1]
    for (S_0, Y_0) in SY_0
        D_0 = sum([ F[m](S_0) for m in 1:length(F) ] .* [ G[n](Y_0) for n in 1:length(G) ])
        S_t = S_0
        Y_t = Y_0
        S_t_prev = S_t
        Y_t_prev = Y_t
        D_t = D_0

        D_best_iter_alt = Inf
        D_best_iter_prev_alt = Inf
        S_best_iter_alt = S_t
        Y_best_iter_alt = Y_t
        iter = 0
        while iter == 0 || abs(D_best_iter_prev_alt - D_best_iter_alt) >= epsilon # Outer loop for alternating method
            D_hat_t = Inf
            delta = D_t/2
            div_ctr = 0
            dim_val = 2
            slow_ctr = 0
            t = 0
            while t==0 || ( (abs(D_hat_t - D_t) >= epsilon) && (norm(S_t_prev - S_t) >= epsilon_S) )
                D_hat_t = min(D_t, D_hat_t) # If current objective is the minimum so far, replace D_hat
                d_S_t = hcat(ThreadsX.collect(grad_S_F[m](S_t) for m in 1:length(grad_S_F))...) * ThreadsX.collect(G[n](Y_t) for n in 1:length(G)) # Gradient of D w.r.t S evaluated at (Yₜ,Sₜ)
                step_size_S = (D_t - D_hat_t + delta) / (norm(d_S_t)^2) # Polyak step size calculation
                S_step_t = S_t - step_size_S*d_S_t # Take step for S
                S_proj_t, Y_proj_t = projOpt(S_step_t, 0, consts); # Projection (ignore Y_proj_t for power step)
                S_t_prev = S_t # We need to save S^t for the while condition
                S_t = S_proj_t # S^{t+1} = \bar{S}^t
                D_t = ThreadsX.sum( ThreadsX.collect(F[m](S_t) for m in 1:length(F)) .* ThreadsX.collect(G[n](Y_t) for n in 1:length(G)) ) # Calculate the objective for iteration t
                if D_t > D_hat_t # Decrease delta when zigzagging occurs
                    div_ctr += 1
                    if div_ctr==5
                        delta = delta / sqrt(dim_val)
                        dim_val += 1
                        div_ctr = 0
                        slow_ctr = 0
                    end
                elseif D_hat_t - D_t <= 10*epsilon
                    slow_ctr += 1
                    if slow_ctr==5  # Increase delta when convergence is slow
                        delta = delta * sqrt(dim_val)
                        slow_ctr = 0
                        div_ctr = 0
                    end
                end
                t += 1
            end

            # Reset variables when moving to caching optimization round
            D_hat_t = Inf
            delta = D_t/2
            div_ctr = 0
            dim_val = 2
            slow_ctr = 0
            t = 0
            while t==0 || ( (abs(D_hat_t - D_t) >= epsilon) && (norm(Y_t_prev - Y_t) >= epsilon_Y) )
                D_hat_t = min(D_t, D_hat_t) # If current objective is the minimum so far, replace D_hat
                d_Y_t = hcat(ThreadsX.collect(subgrad_Y_G[m](Y_t) for m in 1:length(subgrad_Y_G))...) * ThreadsX.collect(F[n](S_t) for n in 1:length(F)) # Subgradient of D w.r.t Y evaluated at (Y_t,S_t)
                step_size_Y = (D_t - D_hat_t + delta) / (norm(d_Y_t)^2) # Polyak step size calculation
                Y_step_t = Y_t - step_size_Y*d_Y_t # Take step for Y
                S_proj_t, Y_proj_t = projOpt(0, Y_step_t, consts); # Projection (ignore S_proj_t for caching step)
                Y_t_prev = Y_t # We need to save Y^t for the while condition
                Y_t = Y_proj_t # Y^{t+1} = \bar{Y}^t
                D_t = ThreadsX.sum( ThreadsX.collect(F[m](S_t) for m in 1:length(F)) .* ThreadsX.collect(G[n](Y_t) for n in 1:length(G)) ) # Calculate the objective for iteration t
                if D_t > D_hat_t # Decrease delta when zigzagging occurs
                    div_ctr += 1
                    if div_ctr==5
                        delta = delta / sqrt(dim_val)
                        dim_val += 1
                        div_ctr = 0
                        slow_ctr = 0
                    end
                elseif D_hat_t - D_t <= 10*epsilon
                    slow_ctr += 1
                    if slow_ctr==5  # Increase delta when convergence is slow
                        delta = delta * sqrt(dim_val)
                        slow_ctr = 0
                        div_ctr = 0
                    end
                end
                t += 1
            end

            D_best_iter_prev_alt = D_best_iter_alt
            if D_t < D_best_iter_alt # Set new best point for this initial point
                D_best_iter_alt = D_t
                S_best_iter_alt = S_t
                Y_best_iter_alt = Y_t
            end
            iter += 1
        end
        if D_best_iter_alt < D_best_alt # Set new best point out of all initial points
            D_best_alt = D_best_iter_alt
            S_best_alt = S_t
            Y_best_alt = Y_t
        end
    end
    return (D_best_alt, S_best_alt, Y_best_alt)
end

function altMethodMult(SY_0, funcs, consts)
    numof_initpoints = length(SY_0)
    F = funcs.F
    grad_S_F = funcs.grad_S_F
    G = funcs.G
    subgrad_Y_G = funcs.subgrad_Y_G

    D_alt_arr = Array{Float64}(undef, numof_initpoints)
    S_alt_arr = Array{Array{Float64,1}}(undef, numof_initpoints)
    Y_alt_arr = Array{Array{Float64,1}}(undef, numof_initpoints)
    Threads.@threads for (i, (S_0, Y_0)) in collect(enumerate(SY_0))
        D_0 = sum([ F[m](S_0) for m in 1:length(F) ] .* [ G[n](Y_0) for n in 1:length(G) ])
        S_t = S_0
        Y_t = Y_0
        S_t_prev = S_t
        Y_t_prev = Y_t
        D_t = D_0

        D_best_iter_alt = Inf
        D_best_iter_prev_alt = Inf
        S_best_iter_alt = S_t
        Y_best_iter_alt = Y_t
        iter = 0
        while iter == 0 || abs(D_best_iter_prev_alt - D_best_iter_alt) >= epsilon # Outer loop for alternating method
            D_hat_t = Inf
            delta = D_t/2
            div_ctr = 0
            dim_val = 2
            slow_ctr = 0
            t = 0
            while t==0 || ( (abs(D_hat_t - D_t) >= epsilon) && (norm(S_t_prev - S_t) >= epsilon_S) )
                D_hat_t = min(D_t, D_hat_t) # If current objective is the minimum so far, replace D_hat
                d_S_t = hcat([ grad_S_F[m](S_t) for m in 1:length(grad_S_F) ]...) * [ G[n](Y_t) for n in 1:length(G) ] # Gradient of D w.r.t S evaluated at (Yₜ,Sₜ)
                step_size_S = (D_t - D_hat_t + delta) / (norm(d_S_t)^2) # Polyak step size calculation
                S_step_t = S_t - step_size_S*d_S_t # Take step for S
                S_proj_t, Y_proj_t = projOpt(S_step_t, Y_t, consts); # Projection (ignore Y_proj_t for power step)
                S_t_prev = S_t # We need to save S^t for the while condition
                S_t = S_proj_t # S^{t+1} = \bar{S}^t
                D_t = sum([ F[m](S_t) for m in 1:length(F) ] .* [ G[n](Y_t) for n in 1:length(G) ]) # Calculate the objective for iteration t
                if D_t > D_hat_t # Decrease delta when zigzagging occurs
                    div_ctr += 1
                    if div_ctr==5
                        delta = delta / sqrt(dim_val)
                        dim_val += 1
                        div_ctr = 0
                        slow_ctr = 0
                    end
                elseif D_hat_t - D_t <= 10*epsilon
                    slow_ctr += 1
                    if slow_ctr==5  # Increase delta when convergence is slow
                        delta = delta * sqrt(dim_val)
                        slow_ctr = 0
                        div_ctr = 0
                    end
                end
                t += 1
            end

            # Reset variables when moving to caching optimization round
            D_hat_t = Inf
            delta = D_t/2
            div_ctr = 0
            dim_val = 2
            slow_ctr = 0
            t = 0
            while t==0 || ( (abs(D_hat_t - D_t) >= epsilon) && (norm(Y_t_prev - Y_t) >= epsilon_Y) )
                D_hat_t = min(D_t, D_hat_t) # If current objective is the minimum so far, replace D_hat
                d_Y_t = hcat([ subgrad_Y_G[m](Y_t) for m in 1:length(subgrad_Y_G) ]...) * ([ F[n](S_t) for n in 1:length(F) ]) # Subgradient of D w.r.t Y evaluated at (Y_t,S_t)
                step_size_Y = (D_t - D_hat_t + delta) / (norm(d_Y_t)^2) # Polyak step size calculation
                Y_step_t = Y_t - step_size_Y*d_Y_t # Take step for Y
                S_proj_t, Y_proj_t = projOpt(S_t, Y_step_t, consts); # Projection (ignore S_proj_t for caching step)
                Y_t_prev = Y_t # We need to save Y^t for the while condition
                Y_t = Y_proj_t # Y^{t+1} = \bar{Y}^t
                D_t = sum([ F[m](S_t) for m in 1:length(F) ] .* [ G[n](Y_t) for n in 1:length(G) ]) # Calculate the objective for iteration t
                if D_t > D_hat_t # Decrease delta when zigzagging occurs
                    div_ctr += 1
                    if div_ctr==5
                        delta = delta / sqrt(dim_val)
                        dim_val += 1
                        div_ctr = 0
                        slow_ctr = 0
                    end
                elseif D_hat_t - D_t <= 10*epsilon
                    slow_ctr += 1
                    if slow_ctr==5  # Increase delta when convergence is slow
                        delta = delta * sqrt(dim_val)
                        slow_ctr = 0
                        div_ctr = 0
                    end
                end
                t += 1
            end
            D_best_iter_prev_alt = D_best_iter_alt
            if D_t < D_best_iter_alt # Set new best point for this initial point
                D_best_iter_alt = D_t
                S_best_iter_alt = S_t
                Y_best_iter_alt = Y_t
            end
            iter += 1
        end
        D_alt_arr[i] = D_best_iter_alt
        S_alt_arr[i] = S_best_iter_alt
        Y_alt_arr[i] = Y_best_iter_alt
    end
    D_best_alt = minimum(D_alt_arr)
    S_best_alt = S_alt_arr[argmin(D_alt_arr)]
    Y_best_alt = Y_alt_arr[argmin(D_alt_arr)]
    return (D_best_alt, S_best_alt, Y_best_alt)
end