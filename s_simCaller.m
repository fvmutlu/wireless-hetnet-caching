% This is the caller script for simulations
clear variables; close all; clc; % To clean up between runs

% YOU CAN CHANGE VARIABLES BELOW THIS LINE, BE SURE TO CHECK THE COMMENTS
V = 21; % Total number of nodes in the topology
SC = 7; % Total number of small cells in the topology
M = 4; % Total number of items (catalog size)
R_cell = 5; % Radius of the cell
plots_enabled = 1; % Remember to make this 0 when averaging over multiple topologies!
pathloss_exp = 4; % Pathloss exponent, you can change this (make sure it is at least 2, and preferably at most 4)
P_max = 100; % Total power resource constraint
P_min = 1; % Minimum per-transmitter-node power, FOR SAFE OPERATION MAKE SURE P_min*SC*10 <= P_max
noise = 1; % Constant noise in the network
c_sc = 1; % Cache capacity of small cell nodes (MAKE SURE c_sc <= M and also make sure to get a reasonable balance between this and M for meaningful results)
c_mc = 2; % Cache capacity of macro cell node (MAKE SURE c_mc <= M and also make sure to get a reasonable balance between this and M for meaningful results)
base_lambda = 1; % Base value for request rates, request rate for a request is determined by adding the Zipf distribution probability value corresponding to the requested item to this base (i.e. more popular items will have slightly higher request rates)
C_sc = R_cell^pathloss_exp; % Backhaul link cost for small cells (You can play around with this, but this can affect topology generation and lead to undesirable results)
% YOU SHOULDN'T CHANGE ANYTHING BELOW THIS LINE

U = V-SC-1; % Total number of users (and hence the requests and paths) in the topology
R = U;
C_mc = 0.5*C_sc; % Backhaul link cost for macro cell, half of small cell cost
SINR_sc = ((1/C_sc)*P_max*0.8)/noise;
SINR_mc = ((1/C_mc)*P_max)/noise;
D_bh_sc = 1/log(1+SINR_sc);
D_bh_mc = 1/log(1+SINR_mc);

[V_pos,SC_pos,v1_routing,v2_routing,edges_routing,edges_all,distances_routing,distances_all,u_nodes,sc_nodes,u_sc_assoc] = cellDistribution(V,SC,U,R_cell);
[gains,costs,G,paths,pathcosts,edgepaths] = graphConstruction(pathloss_exp,distances_routing,distances_all,v1_routing,v2_routing,C_sc,C_mc,U,V,SC,R,u_nodes,edges_routing,edges_all,R_cell,V_pos,SC_pos,plots_enabled);

%% Parameter Setup
% Topology
topology.V = V;
topology.V_pos = V_pos;
topology.sc_nodes = sc_nodes;
topology.pathloss_exp = pathloss_exp;
topology.noise = noise;
topology.paths = cellfun(@invPath,paths,'UniformOutput',false); % This is required right now because the topology creating functions return paths in reverse order and with BH node at the end
topology.D_bh_mc = D_bh_mc;
topology.D_bh_sc = D_bh_sc;

% Problem
problem.c_sc = c_sc;
problem.c_mc = c_mc;
problem.M = M;
problem.base_lambda = base_lambda;
problem.P_min = P_min;
problem.P_max = P_max;

%% Call subgradient method
[opt_delay, opt_cache, opt_power] = mainSubgradient(topology,problem);
opt_cache = reshape(opt_cache,[M,V]);
disp(['Optimal value for D^0 is ' num2str(opt_delay)]);
disp('Optimal caching distribution is:');
disp(opt_cache);
disp('Optimal power distribution is:');
disp(opt_power');

function [path_out] = invPath(path_in)
    path_out = path_in(2:end);
    path_out = flip(path_out);
end