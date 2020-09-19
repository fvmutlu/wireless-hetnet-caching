function [armijo_step] = armijoArc(NS, NY, f, f_grad_at_xk, x, xk, constraint_set)
    s_bar = 20;
    beta = 0.1;
    sigma = 0.1;
    bool = 1;
    m = 0;
    f_at_xk = double(subs(f,x,xk));
    while(bool)
        xk_temp = xk - beta^m*s_bar*f_grad_at_xk;
        xk_proj = proj(NS, NY, xk_temp, constraint_set);
        cond = f_at_xk - double(subs(f,x,xk_proj)) >= sigma * f_grad_at_xk * (xk - xk_proj)';
        if (cond)
            bool = 0;
        else
            m = m+1;
        end
    end
    armijo_step = beta^m * s_bar;
end