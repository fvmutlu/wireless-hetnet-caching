function [S_0,Y_0] = randomInitialPoint(dim_S,dim_Y,P_min,P_max,C,cache_capacity,weight) % REMOVE AP FOR TOTAL BUDGET
    mean = (P_max + P_min)/2 + (P_max + P_min)*weight/2;
    var = (P_max + P_min)/10;
    pd = makedist('Normal','mu',mean,'sigma',var);
    S_0 = random(pd,dim_S,1);
    mean = 0.5 + 0.5*weight/2;
    var = 0.1;
    pd = makedist('Normal','mu',mean,'sigma',var);
    Y_0 = random(pd,dim_Y,1);
    [S_0, Y_0] = projOpt(S_0,Y_0,dim_S,dim_Y,P_min,P_max,C,cache_capacity); % REMOVE AP FOR TOTAL BUDGET
end