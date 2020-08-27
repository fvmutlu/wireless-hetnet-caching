% 4/19/2019
% FARUK VOLKAN MUTLU

function [gains,costs,G,paths,pathcosts,edgepaths] = graphConstruction(alpha,distances_routing,distances_all,v1_routing,v2_routing,C_sc,C_mc,U,V,SC,R,u_nodes,edges_routing,edges_all,R_cell,V_pos,SC_pos,plots_enabled)
    gains = distances_all.^(-alpha);
    costs = [max(1,distances_routing.^alpha) C_sc*ones(1,SC) C_mc]; % calculated for edges involved only in routing, not interferences
    G = digraph(v1_routing,v2_routing,costs);
    
    paths = cell(1,U); % Cell for holding resulting shortest paths
    pathcosts = zeros(1,U);
    
    for u=1:U 
        [paths{u},pathcosts(u)] = shortestpath(G,V+1,u_nodes(u)); % backhaul node is node V+1 so all paths are to that node
    end
    
    edgepaths = cell(1,numel(paths));
    
    for x=1:numel(paths)
        for y=1:numel(paths{x})-1
            e1=find(edges_routing(:,1)==paths{x}(y));
            e2=find(edges_routing(e1,2)==paths{x}(y+1));
            if isempty(e2)
                e1=find(edges_routing(:,1)==paths{x}(y+1));
                e2=find(edges_routing(e1,2)==paths{x}(y));
            end
            edgepaths{x}(1,y) = e1(e2);
        end
    end
    
    pos = [950,200,500,380];

    %TO DO: Extend graph preferences functionality
    if(plots_enabled)
        figure('Position',pos);
        topology_plot = scatter(V_pos(:,1),V_pos(:,2),'filled');
        hold on
        scatter(SC_pos(:,1),SC_pos(:,2),'filled');
        th = 0:pi/50:2*pi;
        xunit = R_cell * cos(th);
        yunit = R_cell * sin(th);
        plot(xunit, yunit);
        hold off;
        axis([-R_cell R_cell -R_cell R_cell]);
        pbaspect([1 1 1])

        figure('Position',pos); % Graph plot with paths    
        graph_plot = plot(G,'XData',[V_pos(:,1);R_cell],'YData',[V_pos(:,2);R_cell],'EdgeColor','w');
        hold on
        scatter(SC_pos(:,1),SC_pos(:,2),'filled');
        axis([-R_cell R_cell -R_cell R_cell]);
        pbaspect([1 1 1])        
        for x=1:R
            highlight(graph_plot,paths{x},'EdgeColor','b','LineWidth',2.0);
        end
    end
end