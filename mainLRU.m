function [LRU] = mainLRU(topology,problem)
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
    
    % For later power control fmincon
    optoptions = optimoptions('fmincon','Display','off');

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
    
    %% Form symbolic S and SINR expressions
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
    %% LRU
    MAX_TIME = 100;
    gammar = 0.1; % Zipf distribution constant
    pr = (1./(1:M).^gammar)/sum(1./(1:M).^gammar); % Zipf probability distribution
    S_0 = randomInitialPowerPoint(total_edges,P_min,P_max,0);
    S_t = S_0;
    cache = zeros(M,V);
    for v=1:V
        cache(1:cache_capacity(v),v) = 1;
    end
    LRU = ones(M,V); % Since we AND by cache indicators we don't want times to have 0, so we assume we start from t=2 and have last access timestamps of items as 1
    D_LRU = 0;
    requested_items = zeros(1,total_paths);
    request_rates = zeros(1,total_paths);
    
    for t=2:1:MAX_TIME+1 % Count time slots until MAX_TIME
        LRU_temp = LRU.*cache; % last access timestamps of items still in caches
        LRU_temp(LRU_temp<=0) = Inf; % remove items not cached, if there are Inf min values there's a bug possibly with initial values
        [~,evictions] = min(flip(LRU_temp)); % LRU items that are still in cache is next eviction, get indices from flipped vector so that least popular item will be given in cases of time collision
        evictions = (M+1)-evictions; % get true indices
        % Generate new requests        
        F = sym('f',[1 total_hops]);
        Gintegral = zeros(1,total_hops);
        marker = 0;
        for n=1:total_paths % Update caches and LRU status of all nodes with newly generated items for time slot t
            i = discreteSample(pr,1); % Pick which item is being requested for this path
            requested_items(n) = i; % Item i requested by nth request
            lambda_i_p = base_lambda + pr(i);
            request_rates(n) = lambda_i_p;
            p = paths{n};
            plen = length(p);
            %% Form symbolic F expressions and also calculate numeric G's for time slot t requests, with t-1 information
            for k=1:plen-1 
                pk = p(k);
                pk_next = p(k+1); % pk_next = p_{k+1}
                [~,~,edge] = intersect([pk_next pk],edges,'rows'); % get edge index for (p_{k+1},p_k)
                F(marker+k) = lambda_i_p * (1/log(1+SINR(edge)));
                Gintegral(marker+k) = 1;
                for l=1:k
                    pl = p(l);
                    Gintegral(marker+k) = Gintegral(marker+k) * (1-cache(i,pl));
                end
            end
            p_end = p(plen);
            if (V_pos(p_end,3)==0)
                F(marker+plen) = lambda_i_p*D_bh_mc;
            elseif (V_pos(p_end,3)==1)
                F(marker+plen) = lambda_i_p*D_bh_sc;
            else
                error('Error: a path seems to end with a user node');
            end
            Gintegral(marker+plen) = 1;
            for l=1:plen
                pl = p(l);
                Gintegral(marker+plen) = Gintegral(marker+plen) * (1-cache(i,pl));
            end
            marker = marker + plen;
        end
        Dlru = F * Gintegral';
        Dlru = matlabFunction(Dlru,'vars',{sort(symvar(S))});
        D_LRU = D_LRU + Dlru(S_t'); % Delay for t requests with (t-1) slot policy
        % disp(D_LRU);
        %% Power control with t information
        [S_t_next,~] = fmincon(Dlru,S_t',ones(1,total_edges),P_max,[],[],P_min*ones(total_edges,1),Inf(total_edges,1),[],optoptions);
        S_t = S_t_next';
        %% LRU update loop with t information
        for n=1:total_paths
            p = paths{n};
            plen = length(p);
            i = requested_items(n);
            for k=2:plen % Skip the user node
                pk = p(k);
                LRU(i,pk) = t;
                if(cache(i,pk)==0) % If item was not cached
                    if(sum(cache(:,pk)) >= cache_capacity(pk)) % If there's no space
                        if(cache(evictions(pk),pk)==1) % and LRU item still there, evict it and cache new item
                            cache(evictions(pk),pk)=0;
                        else % if LRU item is not there, an eviction must have already took place for this node during time slot t
                            evict_temp = LRU(:,pk).*cache(:,pk); % last access timestamps of items still in pk cache
                            evict_temp(evict_temp<=0) = Inf; % remove items not cached if there are Inf min values there's a bug possibly with initial values
                            [~,eviction] = min(flip(evict_temp)); % get index of item to be evicted as per LRU, from flipped vector so that least popular item will be given in cases of time collision
                            eviction = (M+1)-eviction; % get true index of that item
                            cache(eviction,pk)=0;
                        end
                    end
                    cache(i,pk)=1;
                else % If item was cached
                    break; % Item will be served from this node, no need to update anything uplink
                end
            end
        end
    end
    disp(D_LRU/MAX_TIME);
end