% Testing subgradient projection method on a simple topology
%      o -> MC
%      |
%      * -> SC
%     / \
% U1 .   . U2
clear all; close all; clc;

%% Standard gradient projection
% No caching, calculate upper bound

noise = 1; % Noise power
lambda1 = 1; % Request rate parameter for U1
lambda2 = 1.5; % Request rate parameter for U2
dir_step = 4; % Constant stepsize (s^k in Bertsekas)
iter_step = 1; % (alpha^k in Bertsekas)
Pmax = 10; % Maximum power per tx
Pmin = 0; % Minimum power per tx

syms s1 s2 s3 % s1: SC-U1 tx power, s2: SC-U2 tx power, s3: MC-SC tx power

x1 = s1/(noise + s2 + s3/16); % SINR for U1
x2 = s2/(noise + s1 + s3/16); % SINR for U2
x3 = s3/noise;                % SINR for SC

f1 = 1/log2(1+x1);            % Delay at SC-U1 hop
f2 = 1/log2(1+x2);            % Delay at SC-U2 hop
f3 = 1/log2(1+x3);            % Delay at MC-SC hop

D = lambda1*(f1+f3) + lambda2*(f2+f3); % Objective function

grad = jacobian(D,[s1; s2; s3]); % Transpose of gradient with respect to powers

s = [1, 1, 1]; % Set initial powers
K = 300; % Set maximum # of iterations
k = 1; % Start from first iteration
n_grad = double(subs(grad,[s1 s2 s3], s)); % Calculate first gradient (needed for first iteration condition check)

while(k<K && norm(n_grad)>1e-4) % Terminate when max # of iterations reached, or gradient is sufficiently small
    n_D = double(subs(D,[s1 s2 s3], s)); % Evaluate objective function
    disp(['Iteration ', num2str(k), ' objective value: ', num2str(n_D)]); % Print iteration objective value
    n_grad = double(subs(grad,[s1 s2 s3], s)); % Calculate gradient
    s_proj = s - dir_step*n_grad; % Take step, obtain power vector before projection
    s_proj(s_proj<Pmin) = Pmin; % Projection onto constraint set
    s_proj(s_proj>Pmax) = Pmax; % Projection onto constraint set
    s = s + iter_step * (s_proj - s); % Calculate next iteration power vector
    k = k+1; % Increase iteration counter
end

%% Subgradient projection
% Caching, SC caches one file out of two, MC caches both, U1 requests file
% 1 and U2 requests file 2

noise = 1; % Noise power
lambda1 = 1; % Request rate parameter for U1
lambda2 = 1.5; % Request rate parameter for U2
cache_size = 1; % Cache size at SC
dir_step = 4; % Constant stepsize (s^k in Bertsekas)
iter_step = 1; % (alpha^k in Bertsekas)
Pmax = 10; % Maximum power per tx
Pmin = 0; % Minimum power per tx

syms s1 s2 s3 % s1: SC-U1 tx power, s2: SC-U2 tx power, s3: MC-SC tx power
syms y_sc1 y_sc2 % y_sc1: Caching variable at SC for file 1, y_sc2: Caching variable at SC for file 2

x1 = s1/(noise + s2 + s3/16); % SINR for U1
x2 = s2/(noise + s1 + s3/16); % SINR for U2
x3 = s3/noise;                % SINR for SC

f1 = 1/log2(1+x1);            % Delay at SC-U1 hop
f2 = 1/log2(1+x2);            % Delay at SC-U2 hop
f3 = 1/log2(1+x3);            % Delay at MC-SC hop

D = lambda1*(f1+f3*(1-y_sc1)) + lambda2*(f2+f3*(1-y_sc2)); % Objective function

grad = jacobian(D,[s1; s2; s3; y_sc1; y_sc2]);
%grad_s = jacobian(D,[s1; s2; s3]); % Transpose of gradient with respect to powers
%grad_y = jacobian(D,[y_sc1; y_sc2]); % Transpose of gradient with respect to caching variables

s = [1 1 1]; % Set initial powers
y = [0.5 0.5]; % Set initial caching
K = 300; % Set maximum # of iterations
k = 1; % Start from first iteration
n_grad = double(subs(grad, [s1 s2 s3 y_sc1 y_sc2], [s y])); % Calculate first gradient (needed for first iteration condition check)
n_grad_s = n_grad(1:length(s));
n_grad_y = n_grad(length(s)+1:length(s)+length(y));

while(k<K && norm(n_grad_s)>1e-4 && norm(n_grad_y)>1e-4) % Terminate when max # of iterations reached, or gradient is sufficiently small
    n_D = double(subs(D,[s1 s2 s3 y_sc1 y_sc2], [s y])); % Evaluate objective function
    disp(['Iteration ', num2str(k), ' objective value: ', num2str(n_D)]); % Print iteration objective value
    
    % Power vector projection
    n_grad = double(subs(grad, [s1 s2 s3 y_sc1 y_sc2], [s y]));
    n_grad_s = n_grad(1:length(s));
    n_grad_y = n_grad(length(s)+1:length(s)+length(y));
    s_proj = s - dir_step*n_grad_s; % Take step, obtain power vector before projection
    s_proj(s_proj<Pmin) = Pmin; % Projection onto constraint set
    s_proj(s_proj>Pmax) = Pmax; % Projection onto constraint set
    s = s + iter_step * (s_proj - s); % Calculate next iteration power vector
    
    % Caching vector projection
    y_proj = y - dir_step*n_grad_y; % Take step, obtain caching vector before projection
    if (sum(y_proj)>cache_size)
        y_proj = y_proj./sum(y_proj); % Projection onto constraint set (cache size constraint, normalize so sum equals cache size)
    end
    y_proj(y_proj<0) = 0; % Projection onto constraint set (cache integrality constraint, relaxed)
    y_proj(y_proj>1) = 1; % Projection onto constraint set (cache integrality constraint, relaxed)
    y = y + iter_step * (y_proj - y); % Calculate next iteration caching vector
    k = k+1; % Increase iteration counter
end