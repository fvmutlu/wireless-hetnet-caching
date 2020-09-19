function [xk_proj] = proj(NS, NY, xk_temp, constraint_set)
    constraint_set_s = constraint_set{1};
    constraint_set_y = constraint_set{2};
    s_temp = xk_temp(1:NS);
    y_temp = xk_temp(NS+1:NS+NY);
    
    for i=1:length(constraint_set_s)
        s_temp = constraint_set_s{i}(s_temp);
    end
    xk_proj(1:NS) = s_temp;
    
    for i=1:length(constraint_set_y)
        y_temp = constraint_set_y{i}(y_temp);
    end
    xk_proj(NS+1:NS+NY) = y_temp;
end