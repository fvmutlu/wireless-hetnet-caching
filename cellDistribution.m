% 4/19/2019
% FARUK VOLKAN MUTLU

function [V_pos,SC_pos,v1_routing,v2_routing,edges_routing,edges_all,distances_routing,distances_all,u_nodes,sc_nodes,u_sc_assoc] = cellDistribution(V,SC,U,R_cell)
    V_pos = zeros(V,3); % Last column is MC=0,SC=1,U=2 identifier
    SC_pos = zeros(SC,2); % Doesn't need identifier since all of this is SCs
    R_sc = R_cell/2; % Initial coverage area for small cells
    
    for v=2:V
        norm = R_cell+1; %+1 is arbitrary, I just needed something larger than R_cell for convenience of the following loop because matlab doesn't have do..while
        V_pos(v,:) = [0 0 0]; %(0,0,0) is the MC entry hence further nodes shouldn't be placed there
        while (norm>R_cell) || ((V_pos(v,1)==0) && (V_pos(v,2)==0)) %if point outside cell area or is at the center find new point
            V_pos(v,1) = 2*R_cell*rand - R_cell;
            V_pos(v,2) = 2*R_cell*rand - R_cell;
            V_pos(v,3) = 2; % mark node as U
            norm = sqrt(V_pos(v,1)^2+V_pos(v,2)^2);
        end
    end

    sc_nodes = zeros(1,SC);
    %assign random node to be first small cell (TO DO: add away from macro cell condition)
    rand_node = round((V-2)*rand)+2;
    sc_nodes(1) = rand_node;
    SC_pos(1,:) = V_pos(rand_node,1:2);
    V_pos(rand_node,3) = 1; % mark node as SC
    sc_count = 1; % since there's now 1 SC, sc_count = 1

    %assign further nodes to small cell set (TO DO: add away from macro cell condition)
    while(sc_count<SC)
        for v=1:V %go through the node set
            if (sc_count<SC) %if there is still need for a SC, get the distance from this node to each SC node
                diff_to_sc(:,1) = SC_pos(:,1) - V_pos(v,1); diff_to_sc(:,2) = SC_pos(:,2) - V_pos(v,2);
                diff_to_sc = diff_to_sc.^2;
                dist_to_sc = sum(diff_to_sc,2);
                dist_to_sc = sqrt(dist_to_sc); %this is the vector of distances to each SC node
                if all(dist_to_sc > R_sc)
                    sc_count = sc_count + 1;
                    sc_nodes(sc_count) = v;
                    SC_pos(sc_count,:) = V_pos(v,1:2);
                    V_pos(v,3) = 1; % mark node as SC                
                end
            end    
        end
        R_sc = R_sc - R_sc/5; % Decrease R_sc to its %80 at each failed iteration 
    end

    u_nodes = (find(V_pos(:,3)==2))';
    mc_dist = sqrt(sum((V_pos(:,1:2).^2),2));
    u_sc_assoc = zeros(1,U);
    sc_u_dist = zeros(SC,U);
    u_sc_dist = zeros(1,U);
    for i=1:U
        diff_to_sc(:,1) = SC_pos(:,1) - V_pos(u_nodes(i),1); diff_to_sc(:,2) = SC_pos(:,2) - V_pos(u_nodes(i),2);
        diff_to_sc = diff_to_sc.^2;
        dist_to_sc = sum(diff_to_sc,2);
        dist_to_sc = sqrt(dist_to_sc);   
        sc_u_dist(:,i) = dist_to_sc; 
        tmp_u_assoc = sc_nodes(find(dist_to_sc==min(dist_to_sc)));
        tmp_u_dist = min(dist_to_sc);
        if (tmp_u_dist < mc_dist(u_nodes(i)))
            u_sc_assoc(i) = tmp_u_assoc;
            u_sc_dist(i) = tmp_u_dist;
        else
            u_sc_assoc(i) = 1;
            u_sc_dist(i) = mc_dist(u_nodes(i));
        end
    end
    sc_u_dist = reshape(sc_u_dist',[1, SC*U]);

    v_temp1 = []; % v1 side of SC-SC edges
    for i=1:SC-1
        v_temp1 = [v_temp1 sc_nodes(i)*ones(1,SC-i)];
    end

    v_temp2 = []; % v2 side of SC-SC edges
    for i=1:SC-1
        v_temp2 = [v_temp2 sc_nodes(i+1:SC)];
    end
    
    v_temp3 = []; % v1 side of SC-U edges
    for i=1:SC
        v_temp3 = [v_temp3 sc_nodes(i)*ones(1,U)];
    end
    
    v_temp4 = []; % v2 side of SC-U edges
    for i=1:SC
        v_temp4 = [v_temp4 u_nodes];
    end
    
    v1_routing = [ones(1,SC)   u_sc_assoc   v_temp1    v_temp2    (V+1)*ones(1,SC+1)];
    v2_routing = [sc_nodes      u_nodes        v_temp2    v_temp1    sc_nodes 1];
    
    v1_all = [ones(1,U)   ones(1,SC)   v_temp3   v_temp1   v_temp2    (V+1)*ones(1,SC+1)];
    v2_all = [u_nodes     sc_nodes      v_temp4   v_temp2   v_temp1    sc_nodes 1];

    edges_routing = [v1_routing' v2_routing']; % [MC-SC edges  SC-U assoc edges  SC-SC edges  BH-SC edges  BH-MC]    
    edges_all = [v1_all' v2_all']; % [MC-U edges  MC-SC edges  SC-U all edges  SC-SC edges  BH-SC edges  BH-MC]

    mc_u_dist = (mc_dist(u_nodes))';
    mc_sc_dist = (mc_dist(sc_nodes))'; % added this for MC-SC edges
    sc_sc_dist = zeros(1,numel(v_temp1));
    for i=1:numel(v_temp1)
        sc_sc_dist(i) = sqrt((V_pos(v_temp1(i),1) - V_pos(v_temp2(i),1))^2 + (V_pos(v_temp1(i),2) - V_pos(v_temp2(i),2))^2);
    end

    distances_routing = [mc_sc_dist u_sc_dist sc_sc_dist sc_sc_dist]; % No distance entries needed for BH links, costs are constant
    distances_all = [mc_u_dist mc_sc_dist sc_u_dist sc_sc_dist sc_sc_dist];
end