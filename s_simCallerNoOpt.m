% Updated: 7/2/2020
% FARUK VOLKAN MUTLU
diary on;
%% Prep
clear all; close all; clc; % Remove all data generated from the previous simulation
plots_enabled = 0; % No topology plots. Keep this 0 unless you know what you're doing and you need to debug something!

%% Topology parameters
R_cell = 5; % Radius of macro cell service area
V = 50; % Total number of nodes excluding backhaul
%V_arr = 30:5:100;
%SC = 7;
SC_arr = 7:1:24; % Range of values for which we will observe behavior

%% Network parameters
M = 10; % Number of files
pathloss_exp = 4;
base_lambda = 1;
NoisePower = 0.1;
NoiseVar = 1; % Variance of noise distribution (What to do with this?)
Pmax = 100; % Maximum value of transmit power?
%Pmax_arr = 10:10:100;
%Ptotal = Pmax*(SC+1);
%fixed_power = 1;
req_per_user = 1; % Requests per user
%R = req_per_user * U; % Total number of requests
c_sc = round(0.2*M); % Small cells cache %20 of the catalog
c_mc = round(0.4*M); % Macro cell caches %40 of the catalog
c_per_arr = 0.05:0.05:0.45;
C_sc = R_cell^pathloss_exp; % Backhaul link cost for small cells (TO DO: adjust this properly)
C_mc = 0.8*C_sc; % Backhaul link cost for macro cell (TO DO: adjust this properly)
D_bh_mc = 1; % BH delay, adjust this
D_bh_sc = 2; % BH delay, adjust this
gammar = 0.1; % Zipf distribution reflecting item popularity for requests

K = 1000; % Number of simulation iterations
D_arr = []; % Array to store resulting average delay values from each iteration
tic; % Start clock to track elapsed time running function
for SC = SC_arr % Loop based on what value's effect we're simulating. TODO: Make this modular
    Ptotal = Pmax*(SC+1);
    U = V-SC-1;
    R = req_per_user * U; % Total number of requests
    D_avg = 0;
    parfor i=1:K % Main iteration loop can be parallelized, note that this will disallow debug mode
    %% Cell distribution
        [V_pos,SC_pos,v1_routing,v2_routing,edges_routing,edges_all,distances_routing,distances_all,u_nodes,sc_nodes,u_sc_assoc] = cellDistribution(V,SC,U,R_cell);
        [gains,costs,G,paths,pathcosts,edgepaths] = graphConstruction(pathloss_exp,distances_routing,distances_all,v1_routing,v2_routing,C_sc,C_mc,U,V,SC,R,u_nodes,edges_routing,edges_all,R_cell,V_pos,SC_pos,plots_enabled);
    %% Caching parameters
        cache_cap = zeros(V,1); % Cache capacities (users have 0)
        cache_cap(1) = c_mc; % for MC node
        cache_cap(sc_nodes) = c_sc; % for SC nodes
        fixed_caching = 1;
        if(fixed_caching==0)
            gammac = gammar/(pathloss_exp/2+1); % Zipf distribution parameter for caching priority
            pc = 1./(1:M).^gammac/sum(1./(1:M).^gammac); % Caching zipf distribution
        elseif(fixed_caching==1)
            caching_dist = zeros(V,M);
            caching_dist(1,1:c_mc) = 1;
            caching_dist(2:SC+1,1:c_sc) = 1;
        end
    %% Power parameters
        fixed_power = 0; % Keep total power in cell constant
    %% Main function call
        [D, actual_paths, SINR_U, SINR_SC, is] = mainNoOpt(M, gammar, base_lambda, V, U, u_nodes, SC, sc_nodes, u_sc_assoc, edges_routing, edges_all, paths, costs, gains, c_sc, c_mc, C_sc, C_mc, D_bh_mc, D_bh_sc, NoisePower, Pmax, Ptotal, fixed_power);
        D_avg = D_avg+D;
    end
    D_avg = D_avg/K;
    D_arr = [D_arr D_avg];
end
toc;
diary off;