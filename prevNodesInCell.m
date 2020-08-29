% Function that finds uplink nodes for a given node ('val'), in a cell consisting of paths ('cell')
% It simply looks through all paths and finds the node that comes before
% 'val' in each of those paths.
% 1/29/2019
% FARUK VOLKAN MUTLU
function prev_nodes = prevNodesInCell(cell, val)
    prev_node_indices = [];
    prev_nodes = [];
    for i=1:numel(cell)
        if (ismember(val, cell{i}) && cell{i}(1)~=val)
            prev_node_indices = [prev_node_indices (find(cell{i}==val)-1)];
            prev_nodes = [prev_nodes cell{i}(find(cell{i}==val)-1)];        
        end
    end
    prev_nodes = unique(prev_nodes,'stable');
end
