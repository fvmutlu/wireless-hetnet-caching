clear all; close all; clc;

noise = 1; % Noise power
lambda1 = 1; % Request rate parameter for U1
lambda2 = 1; % Request rate parameter for U2
cache_size = 1; % Cache size at SC
dir_step = 4; % Constant stepsize (s^k in Bertsekas)
iter_step = 1; % (alpha^k in Bertsekas)
Pmax = 20; % Maximum power per tx
Pmin = 1; % Minimum power per tx
K = 1000; % Set maximum # of iterations

% figure;
% hold on;
%% Standard gradient projection without caching (1 hop)
% Testing gradient projection method on a simple topology
%      o -> MC
%      |
%      * -> SC
%     / \
% U1 .   . U2
%
% No caching, calculate upper bound

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

%% Standard gradient projection with caching (1 hop)
% Testing gradient projection method on a simple topology
%      o -> MC
%      |
%      * -> SC
%     / \
% U1 .   . U2
%
% Caching, SC caches one file out of two, MC caches both, U1 requests file
% 1 and U2 requests file 2

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

s = [1 1 1]; % Set initial powers
y = [0.5 0.5]; % Set initial caching
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

%% Standard gradient projection (?) without caching (2 hops)
% Testing subgradient projection method on a simple topology
%      o -> MC
%      |
%      * -> SC2
%      |
%      * -> SC1
%     / \
% U1 .   . U2

syms s1 s2 s3 s4 % s1: SC1-U1 tx power, s2: SC1-U2 tx power, s3: SC2-SC1 tx power, s4: MC-SC2 tx power

x1 = s1/(noise + s2 + s4/81); % SINR for U1
x2 = s2/(noise + s1 + s4/81); % SINR for U2
x3 = s3/(noise + s4/16);      % SINR for SC1
x4 = s4/(noise + s1 + s2);    % SINR for SC2

f1 = 1/log2(1+x1);            % Delay at SC1-U1 hop
f2 = 1/log2(1+x2);            % Delay at SC1-U2 hop
f3 = 1/log2(1+x3);            % Delay at SC2-SC1 hop
f4 = 1/log2(1+x4);            % Delay at MC-SC2 hop

D = lambda1*(f1+f3+f4) + lambda2*(f2+f3+f4); % Objective function

grad = jacobian(D,[s1; s2; s3; s4]); % Transpose of gradient with respect to powers

s = [1, 1, 1, 1]; % Set initial powers
k = 1; % Start from first iteration
n_grad = double(subs(grad,[s1 s2 s3 s4], s)); % Calculate first gradient (needed for first iteration condition check)

while(k<=K && norm(n_grad)>1e-4) % Terminate when max # of iterations reached, or gradient is sufficiently small
    n_D = double(subs(D,[s1 s2 s3 s4], s)); % Evaluate objective function
    disp(['Iteration ', num2str(k), ' objective value: ', num2str(n_D)]); % Print iteration objective value
    n_D_arr(k) = n_D;
    n_grad = double(subs(grad,[s1 s2 s3 s4], s)); % Calculate gradient
    s_proj = s - dir_step*n_grad; % Take step, obtain power vector before projection
    s_proj(s_proj<Pmin) = Pmin; % Projection onto constraint set
    s_proj(s_proj>Pmax) = Pmax; % Projection onto constraint set
    s = s + iter_step * (s_proj - s); % Calculate next iteration power vector
    k = k+1; % Increase iteration counter
end

plot(n_D_arr);

symObj = syms;
cellfun(@clear,symObj)

%% Subgradient projection (?) with caching (2 hops)
% Testing subgradient projection method on a simple topology
%      o -> MC
%      |
%      * -> SC2
%      |
%      * -> SC1
%     / \
% U1 .   . U2

syms s1 s2 s3 s4 % s1: SC1-U1 tx power, s2: SC1-U2 tx power, s3: SC2-SC1 tx power, s4: MC-SC2 tx power
NS = 4;
syms y_sc1_1 y_sc1_2 y_sc2_1 y_sc2_2 % y_sc1: Caching variable at SC for file 1, y_sc2: Caching variable at SC for file 2
NY = 4;

sinr1 = s1/(noise + s2 + s4/81); % SINR for U1
sinr2 = s2/(noise + s1 + s4/81); % SINR for U2
sinr3 = s3/(noise + s4/16);      % SINR for SC1
sinr4 = s4/(noise + s1 + s2);    % SINR for SC2

f1 = 1/log2(1+sinr1);            % Delay at SC1-U1 hop
f2 = 1/log2(1+sinr2);            % Delay at SC1-U2 hop
f3 = 1/log2(1+sinr3);            % Delay at SC2-SC1 hop
f4 = 1/log2(1+sinr4);            % Delay at MC-SC2 hop

g1_1 = 1 - piecewise(y_sc1_1<1, y_sc1_1, 1); % Relaxed caching sum for U1's request at SC1
g1_2 = 1 - piecewise(y_sc1_2<1, y_sc1_2, 1); % Relaxed caching sum for U2's request at SC1
g2_1 = 1 - piecewise(y_sc1_1+y_sc2_1<1, y_sc1_1+y_sc2_1, 1); % Relaxed caching sum for U1's request at SC2
g2_2 = 1 - piecewise(y_sc1_2+y_sc2_2<1, y_sc1_2+y_sc2_2, 1); % Relaxed caching sum for U2's request at SC2

D = lambda1*(f1+f3*g1_1+f4*g2_1) + lambda2*(f2+f3*g1_2+f4*g2_2); % Objective function

grad = jacobian(D,[s1; s2; s3; s4; y_sc1_1; y_sc1_2; y_sc2_1; y_sc2_2]); % Subgradient function

constraint_set_s = {@(s_temp) max(s_temp,Pmin), @(s_temp) min(s_temp,Pmax)};
constraint_set_y = {@(y_temp) [(((sum(y_temp(1:2))>cache_size))*y_temp(1:2).*(cache_size/sum(y_temp(1:2))) + (~(sum(y_temp(1:2))>cache_size))*y_temp(1:2)) ...
                             (((sum(y_temp(3:4))>cache_size))*y_temp(3:4).*(cache_size/sum(y_temp(3:4))) + (~(sum(y_temp(3:4))>cache_size))*y_temp(3:4))], ...
                    @(y_temp) max(y_temp,0), @(y_temp) min(y_temp,1)};
constraint_set = {constraint_set_s, constraint_set_y};

s = [1 1 1 1]; % Set initial powers (length must be NS)
y = [0.5 0.5 0.5 0.5]; % Set initial caching (length must be NY)
k = 1; % Start from first iteration
xk = [s y];

n_grad = double(subs(grad, [s1 s2 s3 s4 y_sc1_1 y_sc1_2 y_sc2_1 y_sc2_2], xk)); % Calculate first gradient (needed for first iteration condition check)

while(k<=K && norm(n_grad)>1e-3) % Terminate when max # of iterations reached, or gradient is sufficiently small
    n_D = double(subs(D, [s1 s2 s3 s4 y_sc1_1 y_sc1_2 y_sc2_1 y_sc2_2], xk)); % Evaluate objective function
    disp(['Iteration ', num2str(k), ' objective value: ', num2str(n_D)]); % Print iteration objective value
    %n_D_arr(k) = n_D;
    n_grad = double(subs(grad, [s1 s2 s3 s4 y_sc1_1 y_sc1_2 y_sc2_1 y_sc2_2], xk));
    dir_step = armijoArc(NS, NY, D, n_grad, [s1 s2 s3 s4 y_sc1_1 y_sc1_2 y_sc2_1 y_sc2_2], xk, constraint_set);
    xk_temp = xk - dir_step*n_grad;
    xk_proj = proj(NS, NY, xk_temp, constraint_set);
    xk = xk + iter_step * (xk_proj - xk);
    k = k+1; % Increase iteration counter
end

%plot(n_D_arr);

symObj = syms;
cellfun(@clear,symObj)
clear symObj