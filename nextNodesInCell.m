% Function that finds downlink nodes for a given node ('val'), in a cell consisting of paths ('cell')
% It simply looks through all paths and finds the node that comes after
% 'val' in each of those paths.
% 1/29/2019
% FARUK VOLKAN MUTLU
function next_nodes = nextNodesInCell(cell, val)
    next_nodes = [];
    for i=1:numel(cell)
        if (ismember(val, cell{i}))
            if(find(cell{i}==val) ~= numel(cell{i}))
                next_nodes = [next_nodes cell{i}(find(cell{i}==val)+1)];
            end
        end
    end
    next_nodes = unique(next_nodes,'stable');
end
