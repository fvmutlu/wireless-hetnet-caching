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
