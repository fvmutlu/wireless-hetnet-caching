% Function that calculates delays for a given topology with fixed caching and fixed power
% Each user makes one request according to the zipf distribution. Each
% SC and MC caches the most popular contents according to their
% capacity. Possible transmissions are MC-SC and SC-U. Every
% transmission is on the same channel so interference comes from every
% transmitter in the network. Powers are fixed and each transmitter
% spends a certain amount of power to transmit to a downlink request
% (instead of node, since MC-SC transmissions can have multiple requests
% on the same edge). User SINRs and SC SINRs are calculated separately.
% If SCs don't have requested item cached, they will either go to MC or
% BH. They will go to MC in cases (1) and (2) that are explained below.
% Last edited: 8/28/2020
% Faruk Volkan Mutlu
function [D, actual_paths, SINR_U, SINR_SC, is] = mainNoOpt(M, gammar, base_lambda, V, U, u_nodes, SC, sc_nodes, u_sc_assoc, edges_routing, edges_all, paths, costs, gains, c_sc, c_mc, C_sc, C_mc, D_bh_mc, D_bh_sc, NoisePower, Pmax, Ptotal, fixed_power)

   %% Requests (assume each user makes only one request)
    pr = (1./(1:M).^gammar)/sum(1./(1:M).^gammar);
    is = zeros(V,M);
    lambdais = zeros(V,M);
    for u=1:U
        u_node = u_nodes(u);
        item = discreteSample(pr,1);
        lambdais(u_node) = base_lambda*pr(item);
        is(u_node,item) = 1; % mark the item as requested in user's row in requests matrix
        if(item>c_sc)                  % if the item is not popular enough to be in a SC cache
            path_scs = paths{u}(2:numel(paths{u})-1);
            is(path_scs,item) = 1; % mark as requested in all SCs on path (this can mark it on MC but it doesn't matter)
        end
    end
    
   %% Establish downlinks of transmitters
    downlink = cell(1,SC+1);
    downlink_mc = nextNodesInCell(paths,1); % in paths, find which nodes come after MC
    tmp_downlink_mc = setdiff(downlink_mc,sc_nodes);
    sc_downlinks_of_mc = setdiff(downlink_mc,tmp_downlink_mc);
    downlink_mc = tmp_downlink_mc;
    for sc_node=sc_downlinks_of_mc
        [~,~,sc_mc_edge] = intersect([1 sc_node],edges_routing,'rows'); % find the edge between SC and MC
        req_items = find(is(sc_node,:)==1);
        for req_item=req_items
            if((req_item<=c_mc) && (C_sc>costs(sc_mc_edge)))       % if SC has downlink requesting an item found in MC cache
                downlink_mc = [downlink_mc sc_node];                  % count as downlink if SC-BH costs more than MC-SC (1)
            elseif(req_item>c_mc && C_sc>(costs(sc_mc_edge)+C_mc)) % if SC has downlink requesting content not found in MC cache
                downlink_mc = [downlink_mc sc_node];                  % count as downlink if SC-BH costs more than MC-SC+MC-BH (2)
            end
        end
    end
    downlink{1} = downlink_mc;
    downlink_count_arr = numel(downlink_mc);
    
    for sc=1:SC % for SCs
        downlink_sc = nextNodesInCell(paths,sc_nodes(sc)); % downlinks of sc
        sc_downlinks = setdiff(downlink_sc,u_nodes); % sc downlinks of sc
        downlink_sc = setdiff(downlink_sc,sc_nodes); % remove sc downlinks of sc from downlinks of sc, we'll add them depending on conditions in the following
        for node=sc_downlinks % for all SC downlinks of an SC, count it for a number of times equal to the number of requests forwarded to this SC (meaning items with item>c_sc)
            req_items = find(is(node,:)==1);
            downlink_sc = [downlink_sc node*ones(1,numel(req_items))]; % if the downlink SC has users requesting items with item>c_sc, its "is" row was marked, so directly use the number of such marks for requests
        end
        downlink{1+sc} = downlink_sc;
        downlink_count_arr = [downlink_count_arr numel(downlink_sc)];
    end
    
    downlink_count = sum(downlink_count_arr);
    
   %% Powers
   if(fixed_power==0) % Keeping Ptotal constant, power for each downlink transmission is equal
        s = (Ptotal/downlink_count)*ones(downlink_count,1); % Not really needed, kept for potential future stuff
        for j=1:SC+1
            power_totals(j) = downlink_count_arr(j)*(Ptotal/downlink_count);
        end
   elseif(fixed_power==1) % Keeping Pmax constant
        s = zeros(downlink_count,1); % Not really needed, kept for potential future stuff
        % For each SC (or MC for j=1) get the number of downlink nodes and
        % distribute Pmax equally. Total power at each SC (and MC) will be
        % Pmax.    
        for j=1:SC+1
            s_tmp = (Pmax/downlink_count_arr(j))*ones(downlink_count_arr(j),1); % Not really needed, kept for potential future stuff
            s = [s;s_tmp]; % Not really needed, kept for potential future stuff
            power_totals(j) = Pmax;
        end
   end
    
   %% SINR Calculation
    SINR_U = zeros(1,U);
    for u=1:U % SINR for users
        u_node = u_nodes(u); % pick user node
        assoc_sc = u_sc_assoc(u); % find connected SC node
        [~,~,u_assoc_edge] = intersect([assoc_sc u_node],edges_all,'rows'); % find connecting edge
        if(fixed_power==0)
            Pu = gains(u_assoc_edge)*(Ptotal/downlink_count);
        elseif(fixed_power==1)
            if(assoc_sc==1)
                n = 1;
                Pu = gains(u_assoc_edge)*(Pmax/downlink_count_arr(1));
            else
                n = find(sc_nodes==assoc_sc);
                Pu = gains(u_assoc_edge)*(Pmax/downlink_count_arr(1+n));
            end            
        end
        [~,~,interference_edge] = intersect([1 u_node],edges_all,'rows'); % find the edge between MC and U
        interference = gains(interference_edge)*power_totals(1);
        for sc=1:SC
            interferer_sc = sc_nodes(sc);
            [~,~,interference_edge] = intersect([interferer_sc u_node],edges_all,'rows');
            interference = interference + gains(interference_edge)*power_totals(1+sc);
        end
        SINR_U(u) = Pu/(NoisePower + interference - Pu);
    end
    
    SINR_SC = zeros(1,SC);
    for sc=1:SC % SINR for SCs
        sc_node = sc_nodes(sc);
        [~,~,sc_mc_edge] = intersect([1 sc_node],edges_all,'rows'); % find the edge between SC and MC
        interference = gains(sc_mc_edge)*power_totals(1);
        for n=1:SC
            if n~=sc
                interferer_sc = sc_nodes(n);
                [~,~,sc_sc_edge] = intersect([interferer_sc sc_node],edges_all,'rows');
                interference = interference + gains(sc_sc_edge)*power_totals(1+n);
            end
        end
        if (sum(is(sc_node,:))==0)
            SINR_SC(sc) = 0; % If SC can fulfill all requests from cache, it'll be last hop on paths and therefore incur no delay so safe to leave this 0
        elseif(prevNodesInCell(paths,sc_node)==1)
            if(fixed_power==0)
                Psc = gains(sc_mc_edge)*(Ptotal/downlink_count)*sum(downlink_mc==sc_node);
            elseif(fixed_power==1)
                Psc = gains(sc_mc_edge)*(Pmax/downlink_count_arr(1))*sum(downlink_mc==sc_node);
            end
            SINR_SC(sc) = Psc/(NoisePower + interference - Psc);
        elseif(isempty(prevNodesInCell(paths,sc_node)))
            SINR_SC(sc) = 0; % If there is no connection to the SC, it is safe to leave the SINR as zero since these will not be included in delay calculations    
        elseif(prevNodesInCell(paths,sc_node)==V+1)
            SINR_SC(sc) = D_bh_sc;            
        else
            uplink_sc = prevNodesInCell(paths,sc_node);
            uplink_sc_index = find(sc_nodes==uplink_sc);
            [~,~,uplink_sc_edge] = intersect([uplink_sc sc_node],edges_all,'rows');
            if(fixed_power==0)
                Psc = gains(uplink_sc_edge)*(Ptotal/downlink_count)*sum(downlink{1+uplink_sc_index}==sc_node);
            elseif(fixed_power==1)
                Psc = gains(uplink_sc_edge)*(Pmax/downlink_count_arr(1+uplink_sc_index))*sum(downlink{1+uplink_sc_index}==sc_node);
            end
            SINR_SC(sc) = Psc/(NoisePower + interference - Psc);
        end
    end
    %% Determine actual paths of requests
    actual_paths = cell(1,U);
    for u=1:U % Need to set up "actual paths": paths on which data was returned. This can and will be different in many cases than the initial "paths" due to cache distribution.
        u_node = u_nodes(u);
        sc_node = u_sc_assoc(u);
        % actual_paths{u} = u_node;
        req_item = find(is(u_node,:)==1);
        if req_item <= c_sc
            actual_paths{u} = [sc_node u_node];
        elseif req_item <= c_mc && sum(ismember(downlink{1},sc_node))~=0
            actual_paths{u} = paths{u}(2:length(paths{u}));
        else
            actual_paths{u} = paths{u};
        end            
    end
    %% Calculate delays
    delay_paths = zeros(1,U);
    delay_sc = 1./log(1+SINR_SC);
    delay_u = 1./log(1+SINR_U);
    for u=1:U % Calculate delay for each request by adding up delays on each node along the path (start from user node and add delays incurred at each SC node traversed to get to the data)
        for node=actual_paths{u}(2:length(actual_paths{u})-1) % Remove the user since we'll add it later and remove the last hop since it incurs no delay because the file is found there
            if node == 1
                delay_paths(u) = delay_paths(u) + D_bh_mc; % If 2nd to last hop is MC, we need to add BH delay
            else
                sc = find(sc_nodes==node);
                delay_paths(u) = delay_paths(u) + delay_sc(sc);
            end
        end
        delay_paths(u) = delay_paths(u) + delay_u(u);
    end
    D = mean(delay_paths);
end