function [X] = pipageRounding(F,Gintegral,Y_opt,S_opt,M,V,cache_capacity)
    DO_opt = F(S_opt')*transpose(Gintegral(Y_opt'));
    DO_round = DO_opt;
    Y_matrix = reshape(Y_opt,[M,V]);
    for v=1:V
        y = Y_matrix(:,v); % get node v's caching variables
        y(y<1e-3) = 0; % fix floating-point rounding errors
        y(abs(y-1)<1e-3) = 1; % fix floating-point rounding errors
        while (~all(y==0 | y==1))
            y_frac_pair_ind = find(y~=0 & y~=1);
            if(length(y_frac_pair_ind)==1)
                y_temp = y;
                y_temp(y_frac_pair_ind) = 1;
                if(sum(y_temp)<=cache_capacity(v))
                    y(y_frac_pair_ind) = 1;
                else
                    y(y_frac_pair_ind) = 0;
                end
            else
                y_frac_pair_ind = y_frac_pair_ind(1:2);
                y_frac_pair = y(y_frac_pair_ind);
                y1 = y_frac_pair(1);
                y2 = y_frac_pair(2);
                y_temp = y;
                Y_temp = Y_matrix;

                % Case 1: Try rounding y1 to 0
                if(y1+y2<1)
                    y_temp(y_frac_pair_ind) = [0 y1+y2];
                    Y_temp = Y_matrix;
                    Y_temp(:,v) = y_temp;
                    Y_temp = reshape(Y_temp,[M*V,1]);
                    DO_temp = F(S_opt')*transpose(Gintegral(Y_temp'));
                    if (DO_temp <= DO_round)
                        y = y_temp;
                        DO_round = DO_temp;
                    end
                end
                % Case 2: Try rounding y1 to 1
                if(y2-(1-y1)>0)
                    y_temp(y_frac_pair_ind) = [1 y2-(1-y1)];
                    Y_temp = Y_matrix;
                    Y_temp(:,v) = y_temp;
                    Y_temp = reshape(Y_temp,[M*V,1]);
                    DO_temp = F(S_opt')*transpose(Gintegral(Y_temp'));
                    if (DO_temp <= DO_round)
                        y = y_temp;
                        DO_round = DO_temp;
                    end
                end
                % Case 3: Try rounding y2 to 0
                if(y1+y2<1)
                    y_temp(y_frac_pair_ind) = [y1+y2 0];
                    Y_temp = Y_matrix;
                    Y_temp(:,v) = y_temp;
                    Y_temp = reshape(Y_temp,[M*V,1]);
                    DO_temp = F(S_opt')*transpose(Gintegral(Y_temp'));
                    if (DO_temp <= DO_round)
                        y = y_temp;
                        DO_round = DO_temp;
                    end
                end
                % Case 4: Try rounding y2 to 1
                if(y1-(1-y2)>0)
                    y_temp(y_frac_pair_ind) = [y1-(1-y2) 1];
                    Y_temp = Y_matrix;
                    Y_temp(:,v) = y_temp;
                    Y_temp = reshape(Y_temp,[M*V,1]);
                    DO_temp = F(S_opt')*transpose(Gintegral(Y_temp'));
                    if (DO_temp <= DO_round)
                        y = y_temp;
                        DO_round = DO_temp;
                    end
                end
            end
        end
        Y_matrix(:,v) = y;
    end
    X = reshape(Y_matrix,[M*V,1]);
end