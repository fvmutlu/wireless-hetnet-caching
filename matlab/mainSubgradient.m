% This is the main function for the subgradient method
function [DO_best, X, S_best] = mainSubgradient(topology, problem)
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

    %% Establish paths and gains for transmissions and interference
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
    grad_S_F = transpose(jacobian(F,S));
    subgrad_Y_Gprime = transpose(jacobian(Gprime,Y));
    subgrad_Y_G = transpose(A)*subs(subgrad_Y_Gprime,Y,A*Y);
    
    F_ub = F * ones(total_hops,1); % UPPER BOUND TEST
    F_ub = matlabFunction(F_ub,'vars',{sort(symvar(S))}); % UPPER BOUND TEST
    F = matlabFunction(F,'vars',{sort(symvar(S))},'file','F.m');
    G = matlabFunction(G,'vars',{sort(symvar(Y))},'file','G.m');
    Gintegral = matlabFunction(Gintegral,'vars',{sort(symvar(Y))},'file','Gintegral.m');
    grad_S_F = matlabFunction(grad_S_F,'vars',{sort(symvar(S))},'file','grad_S_F.m');
    subgrad_Y_G = matlabFunction(subgrad_Y_G,'vars',{sort(symvar(Y))},'file','subgrad_Y_G.m');

    %% Subgradient method
    N_init = 10; % Number of initial points
    D_best = Inf; % Needed for first loop iteration
    total_iterations = 0; % For convergence rate information
    N_init_counted = 0;
    for i=1:N_init
        % Initialization
        weight = (i-((N_init+1)/2))/(N_init-1);
        [S_0,Y_0] = randomInitialPoint(total_edges,M*V,P_min,P_max,C,cache_capacity,weight);
        optoptions = optimoptions('fmincon','Display','off');
        [~,D_ub] = fmincon(F_ub,S_0',ones(1,total_edges),P_max,[],[],P_min*ones(total_edges,1),Inf(total_edges,1),[],optoptions); % UPPER BOUND TEST
        disp(['Upper bound for initial point ', num2str(i), ' is: ', num2str(D_ub)]); % UPPER BOUND TEST
        D_0 = F(S_0')*transpose(G(Y_0'));
        S_t = S_0;
        Y_t = Y_0;
        D_t = D_0;
        D_hat_t = D_0; % D_hat for Polyak's, keeps track of minimum objective so far

        delta = D_0/2; % delta for Polyak's
        epsilon = 1e-2;
        epsilon_S = 0.5*1e-2; % termination criterion for S, if the (t+1)th iteration's objective value is within epsilon of (t)th iteration, terminate
        epsilon_Y = 0.5*1e-3; % termination criterion for Y, if the (t+1)th iteration's objective value is within epsilon of (t)th iteration, terminate
        div_ctr = 0;
        dim_val = 2;
        slow_ctr = 0;
        slow_val = 2;
        t = 0;

        % Main Loop
        while(t==0 || ((abs(D_hat_t-D_t)>=epsilon) && (norm(Y_t_prev-Y_t)>=epsilon_Y) && (norm(S_t_prev-S_t)>=epsilon_S)))
            % disp(['Iteration ', num2str(t), ' objective value: ', num2str(D_t)]); % Print iteration objective value
            D_hat_t = min(D_t,D_hat_t); % If current objective is the minimum so far, replace D_hat
            d_S_t = grad_S_F(S_t') * transpose(G(Y_t')); % Gradient of D w.r.t S evaluated at (Y_t,S_t)
            d_Y_t = subgrad_Y_G(Y_t') * transpose(F(S_t')); % Subgradient of D w.r.t Y evaluated at (Y_t,S_t)
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
            D_t = F(S_t')*transpose(G(Y_t')); % Calculate the objective for iteration t
            if(D_t > D_hat_t) % Decrease delta when zigzagging occurs
                div_ctr = div_ctr+1;
                if(div_ctr==5)
                    delta = delta/sqrt(dim_val);
                    dim_val = dim_val+1;
                    div_ctr = 0;
                    slow_ctr = 0;
                end
            elseif(abs(D_t-D_hat_t) <= 10*epsilon) % Increase delta when convergence is slow
                slow_ctr = slow_ctr+1;
                if(slow_ctr==5)
                    delta = delta*sqrt(dim_val);
                    slow_val = slow_val+1;
                    slow_ctr = 0;
                    div_ctr = 0;
                end
            end
            t = t+1;
            if(t>150)
                if(D_t > D_best)
                    disp(['Initial point ', num2str(i), ' exceeded 150 iterations, last best value for relaxed problem was: ', num2str(D_hat_t), '. Discarding this initial point.']);
                    break;
                else
                    disp(['Initial point ', num2str(i), ' exceeded 150 iterations, recording last best value for relaxed problem']);
                    t = t-1;
                    break;
                end
            end
        end
        if(t<=150)
            N_init_counted = N_init_counted + 1;
            disp(['Initial point ', num2str(i), ' relaxed problem local optimum value: ', num2str(D_t), '. Found in ', num2str(t), ' iterations.']);
            total_iterations = total_iterations + t;
            if(D_t < D_best)
                D_best = D_t;
                S_best = S_t;
                Y_best = Y_t;
            end
        end
    end
    if(D_best == Inf)
        disp('Bad set of initial points for topology, algorithm did not converge');
        X = zeros(M,V);
        DO_best = 0;
        S_best = zeros(total_edges,1);
    else
        disp(['Best local optimum for relaxed problem found among ', num2str(N_init_counted), ' initial points: ', num2str(D_best)]);
        disp(['Average number of iterations: ', num2str(round(total_iterations/N_init_counted))]);
        % Rounding
        X = pipageRounding(F,Gintegral,Y_best,S_best,M,V,cache_capacity);
        DO_best = F(S_best')*transpose(Gintegral(X'));
        disp(['Best local optimum after rounding: ' num2str(DO_best)]);
    end
end