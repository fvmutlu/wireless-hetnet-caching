% This is the simulation caller for subgradient method
%% Initialize topology parameters
V = 10;
SC = 2;
U = V-SC-1;
R_cell = 5;
M = 5;
pathloss_exp = 4;
base_lambda = 1;
noise = 0.1;
cache_capacity = zeros(V,1); cache_capacity(1) = c_mc; cache_capacity(sc_nodes) = c_sc;

%% Generate randomized topology and establish paths

%% Generate requests assuming each user makes only one request
gammar = 0.1; % Zipf distribution constant
pr = (1./(1:M).^gammar)/sum(1./(1:M).^gammar); % Zipf probability distribution
requests = zeros(V,M);
req_lambda = zeros(V,M);
for u=1:U
    u_node = u_nodes(u); % Get node ID of current user node
    item = discreteSample(pr,1);
    req_lambda(u_node) = base_lambda*pr(item);
    requests(u_node,item) = 1; % mark the item as requested in user's row in requests matrix
    path_scs = paths{u}(2:numel(paths{u})-1);
    requests(path_scs,item) = 1; % mark as requested in all cells on path (SCs and MC) (interest aggregation)
end

%% Form symbolic SINR and f expressions (eq. 1 and 2 in the paper)
S = sym('s',[total_hops 1]);
SINR = sym('sinr',[total_hops 1]);
F = sym('f',[1 total_hops]);

%% Form symbolic caching expressions
Y = sym('y',[M*V 1]); % Here Y is actually vec(Y)^T from the paper
G_prime = sym('gp',[1 total_hops]);
A = zeros(M*V,M*V); % For affine composition G(Y) = G'(AY)
A = [];
%G = sym('g',[1 total_hops]);
G = subs(G_prime,Y,A*Y);
C = zeros(V,M*V); % For cache capacity constraint C*Y <= cache_capacity
for i=1:V
    C(i,(i-1)*M+1:i*M) = 1;
end

%% Form symbolic objective function expression and gradient expressions
D = F*transpose(G);
grad_S_F = transpose(jacobian(F,S));
grad_S_D = grad_S_F * transpose(G);
%subgrad_Y_G = transpose(jacobian(G,Y)); % check if this is the same*
subgrad_Y_G_prime = transpose(jacobian(G_prime,Y));
subgrad_Y_G = transpose(A)*subs(subgrad_Y_G_prime,Y,A*Y);
subgrad_Y_D = F*transpose(subgrad_Y_G);

%% Subgradient method initialization
Y_0 = zeros(M*V,1); % change these
S_0 = zeros(total_hops,1); % change these
D_0 = double(subs(D,[Y;S],[Y_0;S_0]));
D_hat_0 = double(subs(D,[Y;S],[Y_0;S_0])); % D_hat for Polyak's, keeps track of minimum objective so far
delta = 1e-6; % delta for Polyak's, change this

Y_t = Y_0;
S_t = S_0;
D_t = D_0;
Y_t_prev = zeros(M*V,1);
S_t_prev = zeros(total_hops,1);
D_t_prev = D_0;
D_hat_t = D_hat_0;
t = 0;

%% Subgradient method main loop
while(t==0 || Y_t~=Y_t_prev || S_t~=S_t_prev)
    D_t = double(subs(D,[Y;S],[Y_t;S_t])); % Calculate the objective for iteration t
    disp(['Iteration ', num2str(t), ' objective value: ', num2str(D_t)]); % Print iteration objective value
    D_hat_t = min(D_t,D_hat_t); % If current objective is the minimum so far, replace D_hat
    d_S_t = double(subs(grad_S_D,[Y;S],[Y_t;S_t])); % Gradient of D w.r.t S evaluated at (Y_t,S_t)
    d_Y_t = double(subs(subgrad_Y_D,[Y;S],[Y_t;S_t])); % Subgradient of D w.r.t Y evaluated at (Y_t,S_t) *this may require change
    step_size = (D_t - D_hat_t + delta)/(norm(d_S_t)^2+norm(d_Y_t)^2); % Polyak step size calculation
    S_step_t = S_t - step_size*d_S_t; % Take step for S
    Y_step_t = Y_t - step_size*d_Y_t; % Take step for Y
    [S_proj_t, Y_proj_t] = projOpt(S_step_t,Y_step_t,total_hops,M*V,P_min,P_max,C,cache_capacity); % Projection
    S_t_prev = S_t; % We need to save S^t for the while condition
    Y_t_prev = Y_t; % We need to save Y^t for the while condition
    S_t = S_proj_t; % S^{t+1} = \bar{S}^t
    Y_t = Y_proj_t; % Y^{t+1} = \bar{Y}^t
    t = t+1;
end

