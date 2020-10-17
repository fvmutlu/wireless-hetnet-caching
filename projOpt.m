function [S_proj_t, Y_proj_t] = projOpt(S_step_t,Y_step_t,dim_S,dim_Y,P_min,P_max,C,cache_capacity)
% Minimum norm subproblem for S projection
cvx_begin quiet
    variable S_proj_t(dim_S,1)
    minimize(norm(S_proj_t - S_step_t))
    subject to
        S_proj_t >= P_min;
        ones(1,dim_S)*S_proj_t <= P_max;
cvx_end
cvx_clear
% Minimum norm subproblem for Y projection
cvx_begin quiet
    variable Y_proj_t(dim_Y,1)
    minimize(norm(Y_proj_t - Y_step_t))
    subject to
        Y_proj_t >= 0; % Cache integrality constraint lower bound
        Y_proj_t <= 1; % Cache integrality constraint upper bound
        C*Y_proj_t <= cache_capacity; % Cache capacity constraint
cvx_end
cvx_clear
end