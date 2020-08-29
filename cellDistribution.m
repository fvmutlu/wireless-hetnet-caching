% Function that generates nodes within a cell, places them in space and marks their type
% The function places the MC node at the center of the cell area. Then it
% uniformly distributes the remaining nodes within the cell. Afterwards, it
% tries to find suitable nodes to mark as SC nodes so that SCs are distributed
% in a fair manner inside the cell and they don't crowd in a specific
% area. All remaining nodes (that are not marked to be SCs) are left as users.
% Finally, it computes various values needed to build a network graph with
% the given positions of the nodes. The backhaul is not involved in any of
% these distributions/computations since it is assumed to be connected by
% wire and all link costs associated will be constant. The code may
% distinguish between values needed for 'routing' and other computations
% ('all') where needed; this is to reduce the inconvenience of later
% computations when finding edges between certain nodes for path-related
% and interference-related purposes may lead to confusion. Another reason
% for this is to achieve a clean scatter plot of the graph, as explained in
% the graphConstruction function.
% Last edited: 8/28/2020
% Faruk Volkan Mutlu

function [V_pos,SC_pos,v1_routing,v2_routing,edges_routing,edges_all,distances_routing,distances_all,u_nodes,sc_nodes,u_sc_assoc] = cellDistribution(V,SC,U,R_cell)
    V_pos = zeros(V,3); % Last column is node type identifier MC=0,SC=1,U=2
    SC_pos = zeros(SC,2); % Doesn't need identifier since all of this is SCs
    R_sc = R_cell/2; % Initial coverage area for small cells
    
    for v=2:V
        norm = R_cell+1; % +1 is arbitrary, just needs something larger than R_cell for convenience of the following loop
        V_pos(v,:) = [0 0 0]; %(0,0,0) is the MC entry hence further nodes shouldn't be placed there
        while (norm>R_cell) || ((V_pos(v,1)==0) && (V_pos(v,2)==0)) % If point is outside cell area or is at the center find new point
            V_pos(v,1) = 2*R_cell*rand - R_cell; % Generate point in interval (-R_cell, R_cell) and assign to x for node v's position
            V_pos(v,2) = 2*R_cell*rand - R_cell; % Generate point in interval (-R_cell, R_cell) and assign to y for node v's position
            V_pos(v,3) = 2; % Mark node v as User
            norm = sqrt(V_pos(v,1)^2+V_pos(v,2)^2);
        end
    end

    sc_nodes = zeros(1,SC);
    % Assign random node to be first small cell (TODO: add away from macro cell condition)
    rand_node = round((V-2)*rand)+2;
    sc_nodes(1) = rand_node;
    SC_pos(1,:) = V_pos(rand_node,1:2);
    V_pos(rand_node,3) = 1; % Mark node as SC
    sc_count = 1; % since there's now 1 SC, sc_count = 1

    % Assign further nodes to small cell set (TODO: add away from macro cell condition)
    while(sc_count<SC)
        for v=1:V
            if (sc_count<SC) % If there is still need for a SC, get the distance from this node to each SC node
                diff_to_sc(:,1) = SC_pos(:,1) - V_pos(v,1); diff_to_sc(:,2) = SC_pos(:,2) - V_pos(v,2);
                diff_to_sc = diff_to_sc.^2;
                dist_to_sc = sum(diff_to_sc,2);
                dist_to_sc = sqrt(dist_to_sc); % This is the vector of distances from node v to each existing SC node
                if all(dist_to_sc > R_sc) % If all SC nodes are sufficiently far away, node v can be marked as SC
                    sc_count = sc_count + 1;
                    sc_nodes(sc_count) = v;
                    SC_pos(sc_count,:) = V_pos(v,1:2);
                    V_pos(v,3) = 1; % Mark node as SC                
                end
            end    
        end % If for all possible v, some SC nodes were closer than desired, we'll have to adjust expectations
        R_sc = R_sc - R_sc/5; % So decrease R_sc by %20 at each failed iteration
    end

    u_nodes = (find(V_pos(:,3)==2))'; % Vector of nodes that are Users in the final distribution
    mc_dist = sqrt(sum((V_pos(:,1:2).^2),2)); % Vector of distances of each node to MC node
    u_sc_assoc = zeros(1,U); % Vector that contains the node number of closest SC for each User (useful for graph building and various power calculations)
    sc_u_dist = zeros(SC,U); % Matrix that contains distances between users and all SCs (useful for graph building)
    u_sc_dist = zeros(1,U); % Vector that contains distances between users and their closest SCs (useful for graph building)
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
    
    % In below calculations, remember that the node numbered V+1 is the
    % representation of backhaul, or the hypothetical 'backhaul node'
    
    v1_routing = [ones(1,SC)   u_sc_assoc   v_temp1    v_temp2    (V+1)*ones(1,SC+1)]; % [MC-SC edges  SC-U assoc edges  SC-SC edges  BH-SC edges  BH-MC]
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