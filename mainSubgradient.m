% This is the main function for the subgradient method
function [DO_opt] = mainSubgradient(topology, problem)
    %% Initialize parameters
    % Topology
    V = topology.V;
    V_pos = topology.V_pos;
    sc_nodes = topology.sc_nodes;
    pathloss_exp = topology.pathloss_exp;
    noise = topology.noise;
    paths = topology.paths;
    D_bh_mc = topology.D_bh_mc;
    D_bh_sc = topology.D_bh_sc;
    
    % Problem
    cache_capacity = zeros(V,1); cache_capacity(1) = problem.c_mc; cache_capacity(sc_nodes) = problem.c_sc;
    M = problem.M;
    base_lambda = problem.base_lambda;
    P_min = problem.P_min;
    P_max = problem.P_max;

    %% Generate randomized topology and establish paths
    total_paths = length(paths);
    total_hops = sum(cellfun(@length,paths));
    gains = zeros(V,V);
    edges = [];
    for v=1:V
        for u=1:V
            distance = sqrt((V_pos(v,1)-V_pos(u,1))^2+(V_pos(v,2)-V_pos(u,2))^2);
            gains(v,u) = min(1,distance^(-pathloss_exp));
        end
    end
    marker = 0;
    for n=1:total_paths
        p = paths{n};
        plen = length(p);
        for k=1:plen-1
            pk = p(k);
            pk_next = p(k+1);
            edges(marker+k,:) = [pk_next pk];
        end
        marker = marker + plen - 1;
    end
    edges = unique(edges,'rows');
    total_edges = length(edges);

    %% Generate requests assuming each user makes only one request
    gammar = 0.1; % Zipf distribution constant
    pr = (1./(1:M).^gammar)/sum(1./(1:M).^gammar); % Zipf probability distribution
    requested_items = zeros(1,total_paths);
    request_rates = zeros(1,total_paths);
    for n=1:total_paths
        i = discreteSample(pr,1); % Pick which item is being requested for this path
        requested_items(n) = i;
        request_rates(n) = base_lambda + pr(i);
    end

    %% Form symbolic SINR and f expressions (eq. 1 and 2 in the paper)
    S = sym('s',[total_edges 1]);
    SINR = sym('sinr',[total_edges 1]);
    for e=1:total_edges
        edge = e;
        v = edges(edge,1); u = edges(edge,2); % Tx side of edge = v, Rx side = u
        s_vu = S(edge);
        power = gains(v,u)*s_vu;
        interference = 0;
        for edge=1:total_edges
            vprime = edges(edge,1); % get Tx side of edge
            s_vprime_u = S(edge);
            interference = interference + gains(vprime,u)*s_vprime_u;
        end
        interference = interference - power;
        SINR(e) = power/(noise+interference);
    end
    F = sym('f',[1 total_hops]);
    f_marker = 0;
    for n=1:total_paths
        p = paths{n};
        plen = length(p); % plen = |p|
        lambda_i_p = request_rates(n); % request rates holds \lambda_{(i,p)}s
        for k=1:plen-1
            pk = p(k);
            pk_next = p(k+1); % pk_next = p_{k+1}
            [~,~,edge] = intersect([pk_next pk],edges,'rows'); % get edge index for (p_{k+1},p_k)
            F(f_marker+k) = lambda_i_p * (1/log(1+SINR(edge)));
        end
        p_end = p(plen);
        if (V_pos(p_end,3)==0)
            F(f_marker+plen) = lambda_i_p*D_bh_mc;
        elseif (V_pos(p_end,3)==1)
            F(f_marker+plen) = lambda_i_p*D_bh_sc;
        else
            error('Error: a path seems to end with a user node');
        end
        f_marker = f_marker + plen;
    end

    %% Form symbolic caching expressions
    Y = sym('y',[M*V 1]); % Here Y is actually vec(Y)^T from the paper
    Gprime = sym('gp',[1 total_hops]);
    Gintegral = sym('gint',[1 total_hops]); % ROUNDING TEST
    gp_marker = 0; % We need to establish G' functions, since the length of paths are variable we need to keep track of where we are in the following loop
    A = zeros(M*V,M*V); % For affine composition G(Y) = G'(AY)
    for n=1:total_paths % Iterate over all paths (using iterator n) and establish G' functions (length(paths) = length(requests) = U)
        i = requested_items(n); % Item i requested by nth request
        p = paths{n}; % Path p for nth request
        plen = length(p);
        for k=1:plen
            pk = p(k);
            y_pk_i = Y((pk-1)*M + i);
            Gprime(gp_marker+k) = 1 - piecewise(y_pk_i<1, y_pk_i, 1); % Establish symbolic function G' for kth node along path p
            Gintegral(gp_marker+k) = 1;
            for l=1:k
                pl = p(l);
                A((pk-1)*M + i, (pl-1)*M + i) = 1; % In matrix A, mark entries corresponding to all nodes on path p before pk as 1
                y_pl_i = Y((pl-1)*M + i); % ROUNDING TEST
                Gintegral(gp_marker+k) = Gintegral(gp_marker+k) * (1-y_pl_i); % ROUNDING TEST
            end
        end
        gp_marker = gp_marker + plen; % Mark all G' functions related to path p are computed
    end
    G = subs(Gprime,Y,A*Y);
    C = zeros(V,M*V); % For cache capacity constraint C*Y <= cache_capacity
    for n=1:V
        C(n,(n-1)*M+1:n*M) = 1; % In matrix C, mark entries corresponding to node n's items as 1
    end

    %% Form symbolic objective function expression and gradient expressions
    D = F*transpose(G);
    grad_S_F = transpose(jacobian(F,S));
    grad_S_D = grad_S_F * transpose(G);
    subgrad_Y_Gprime = transpose(jacobian(Gprime,Y));
    subgrad_Y_G = transpose(A)*subs(subgrad_Y_Gprime,Y,A*Y);
    subgrad_Y_D = subgrad_Y_G * transpose(F);

    %% Subgradient method initialization
    S_0 = (P_max/total_edges)*ones(total_edges,1); % change this
    %Y_0 = 0.5*ones(M*V,1);
    Y_0 = [1;1;0.9;0.1;0;0;0.1;0.9;0;0]; % change this
    D_0 = double(subs(D,[Y;S],[Y_0;S_0]));
    delta = 4; % delta for Polyak's**
    epsilon = 1e-5; % termination criterion, if the (t+1)th iteration's objective value is within epsilon of (t)th iteration terminate
    S_t = S_0;
    Y_t = Y_0;
    % S_best = S_t;
    % Y_best = Y_t;
    D_t = D_0;
    D_hat_t = D_0; % D_hat for Polyak's, keeps track of minimum objective so far
    t = 0;
    conv_ctr = 0; % TEST
    div_ctr = 2; % TEST
    %% Subgradient method main loop
    while(t==0 || abs(D_t_prev-D_t) >= epsilon)
        disp(['Iteration ', num2str(t), ' objective value: ', num2str(D_t)]); % Print iteration objective value
        D_hat_t = min(D_t,D_hat_t); % If current objective is the minimum so far, replace D_hat
    %     if(D_t<D_hat_t) % TEST
    %         D_hat_t = D_t;
    %         S_best = S_t;
    %         Y_best = Y_t;
    %     end
        d_S_t = double(subs(grad_S_D,[Y;S],[Y_t;S_t])); % Gradient of D w.r.t S evaluated at (Y_t,S_t)
        d_Y_t = double(subs(subgrad_Y_D,[Y;S],[Y_t;S_t])); % Subgradient of D w.r.t Y evaluated at (Y_t,S_t) *this may require change
        step_size_S = (D_t - D_hat_t + delta)/(norm(d_S_t)^2); % Polyak step size calculation
        step_size_Y = (D_t - D_hat_t + delta)/(norm(d_Y_t)^2); % Polyak step size calculation
        S_step_t = S_t - step_size_S*d_S_t; % Take step for S
        Y_step_t = Y_t - step_size_Y*d_Y_t; % Take step for Y
        [S_proj_t, Y_proj_t] = projOpt(S_step_t,Y_step_t,total_edges,M*V,P_min,P_max,C,cache_capacity); % Projection
        S_t_prev = S_t; % We need to save S^t for the while condition
        Y_t_prev = Y_t; % We need to save Y^t for the while condition
        D_t_prev = D_t;
        S_t = S_proj_t; % S^{t+1} = \bar{S}^t
        Y_t = Y_proj_t; % Y^{t+1} = \bar{Y}^t
        D_t = double(subs(D,[Y;S],[Y_t;S_t])); % Calculate the objective for iteration t
        if(D_t > D_hat_t)
            conv_ctr = conv_ctr+1;
            if(conv_ctr==5)
                delta = delta/sqrt(div_ctr);
    %             S_t = S_best;
    %             Y_t = Y_best;
                div_ctr = div_ctr+1;
                conv_ctr = 0;
            end
    %     else
    %         conv_ctr = 0;
    %         delta = delta*2;
        end
        t = t+1;
    end
    S_opt = S_t;
    Y_opt = Y_t;
    D_opt = D_t;

    %% Rounding
    DO = F*transpose(Gintegral);
    X = pipageRounding(DO,Y,S,Y_opt,S_opt,M,V,cache_capacity);
    DO_opt = double(subs(DO,[Y;S],[X;S_opt]));
end