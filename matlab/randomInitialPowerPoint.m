function [S_0] = randomInitialPowerPoint(dim_S,P_min,P_max,weight)
    mean = (P_max + P_min)/2 + (P_max + P_min)*weight;
    var = (P_max + P_min)/10;
    pd = makedist('Normal','mu',mean,'sigma',var);
    S_0 = random(pd,dim_S,1);
end