% Using grid points to simulate 2d Lloyd-Max quantizer

%% Setting
clear all
close all


Welfare_Revenue = 'welfare'; % 'welfare' for welfare maximization, 'revenue' for revenue maximization (only for Separability = 1)

Separability = 'nonseparable'; % 'nonseparable' means quantizing [0,1]x[0,1], 'separable' means quantizing circle with radius 1 in first orthant

Type_Distribution = 'uniform';
% choose from :
...'uniform'(no restrict)(uniform/uniball).
    ...'cone'(only for Separability = 1, Welfare), fx = 2 - 2|x|.
    ...'gaussian'(only for Separability = 1, Welfare/Revenue), truncated Gaussian.
    ...'beta'(only for Separability = 1, Revenue), truncated Beta in Proposition 9.
    
Mu = [0 0]; % mu and sigma for Gaussian distribution
Sigma = [0.2 0;0 0.2];

Max_Iter = 30;
N = 17; % Number of quantization levels

label_offset_x = 0.04;
label_offset_y = 0.03;

%% Generate grid
dx = 2e-3;
dy = dx;

x = dx:dx:1;
y = dx:dy:1;
[x_grid,y_grid] = meshgrid(x,y);

if strcmp(Separability,'nonseparable') == 1 % If quantizing over 2d [0,1]x[0,1]
    Len = length(x) * length(y); % Total number of grid points
    X = zeros(Len,2); % value of (x,y) for each point
    X_pos = zeros(Len,2); % position in x and y for each point
    num = 1;
    for i = 1:1:length(x)
        for j = 1:1:length(y)
            X(num,1) = x(i);
            X(num,2) = y(j);
            X_pos(num,1) = i;
            X_pos(num,2) = j;
            num = num+1;
        end
    end
    
else % If quantizing over quater circle
    num = 1;
    In_Out = x_grid.^2 + y_grid.^2;
    
    Len = sum(sum(In_Out <= 1));
    X = zeros(Len,2);
    X_pos = zeros(Len,2);
    for i = 1:1:length(x)
        for j = 1:1:length(y)
            if(x(i)^2 + y(j)^2 <= 1)
                X(num,1) = x(i);
                X(num,2) = y(j);
                X_pos(num,1) = i;
                X_pos(num,2) = j;
                num = num +1;
            end
        end
    end
    
end

%% Generate distribution

% Set original type distribution
% if Type_Distribution == 'uniform'
% f_uniform = ones(1,Len);
% f_type = f_uniform; % distribution
% end

if strcmp(Welfare_Revenue,'welfare') == 1 % For welfare maximization
    
    if strcmp(Separability,'nonseparable') == 1 % For cubic quantization area
        if strcmp(Type_Distribution,'uniform') == 1 % for cubic uniform distribution welfare maximization
            fprintf('Cubic uniform distribution welfare maximization. Setting...\n');
            f_type = ones(1,Len);
            f = f_type;
            figure(2)
            mesh(x_grid,y_grid,ones(size(x_grid)))
            title('Quantized Distribution')
        else
            fprintf('Unavailable setting!\n');
            return;
        end
        
    elseif strcmp(Separability,'separable') == 1 % For Rotationally invariant distribution
        if strcmp(Type_Distribution,'uniform') == 1 % for seprarable uniform distribution welfare maximization
            fprintf('Seprarable uniball distribution welfare maximization. Setting...\n');
            f_type = ones(1,Len);
            f = f_type;
            figure(2)
            mesh(x_grid,y_grid,ones(size(x_grid)))
            title('Quantized Distribution')
        elseif strcmp(Type_Distribution,'gaussian') == 1 % for truncated Gaussian
            fprintf('Seprarable truncated Gaussian distribution welfare maximization. Setting...\n');
            xy_grid = [x_grid(:) y_grid(:)];
            Gaussian_pdf = mvnpdf(xy_grid,Mu,Sigma);
            Gaussian_pdf = reshape(Gaussian_pdf,length(y),length(x));
            figure(2)
            surf(x,y,Gaussian_pdf);
            f = zeros(1,Len);
            for i = 1:1:Len
                x_pos = X_pos(i,1);
                y_pos = X_pos(i,2);
                f(i) = Gaussian_pdf(x_pos,y_pos);
            end
            figure(2)
            mesh(x_grid,y_grid,Gaussian_pdf)
            title('Quantized Distribution')
        elseif strcmp(Type_Distribution,'cone') == 1 % for cone dist.
            fprintf('Seprarable cone distribution welfare maximization. Setting...\n');
            r = x;
            dr = dx;
            f_type_radial = 2 - 2*x;
            f = zeros(1,Len);
            Cone_pdf = zeros(length(x),length(y));
            for i = 1:1:Len
                rad = sqrt(X(i,1)^2 + X(i,2)^2);
                [temp,r_pos] = min(abs(r - rad));
                f(i) = f_type_radial(r_pos);
                Cone_pdf(X_pos(i,1),X_pos(i,2)) = f_type_radial(r_pos);
            end
            figure(2)
            mesh(x_grid,y_grid,Cone_pdf)
            title('Quantized Distribution')
        else
            fprintf('Unavailable setting!\n');
            return;
        end
    end
    
elseif strcmp(Welfare_Revenue,'revenue') == 1 % For revenue maximization
    % Firts generate the original type radial distribution
    % Then calculate the virtual type radial distribution
    % Finally convert the radial distribution to grid
    
    if strcmp(Separability,'nonseparable') == 1
        fprintf('Unavailable setting!\n');
        return;
    elseif strcmp(Separability,'separable') == 1
        % generate original type radial dist.
        r = x;
        dr = dx;
        if strcmp(Type_Distribution,'uniform') == 1
            fprintf('Seprarable uniball distribution revenue maximization. Setting...\n');
            f_type_radial = ones(size(r));
        elseif strcmp(Type_Distribution,'gaussian') == 1
            fprintf('Seprarable truncated Gaussian distribution revenue maximization. Setting...\n');
            f_type_radial = normpdf(r,Mu(1),Sigma(1,1));
        elseif strcmp(Type_Distribution,'beta') == 1
            fprintf('Seprarable truncated Beta distribution revenue maximization. Setting...\n');
            f_type_radial = r.^(- 1/2);
        else
            fprintf('Unavailable setting!\n');
            return;
        end
        
        % calculate virtual type virtual radial dist.
        f_Vtype_radial = zeros(size(f_type_radial));
        if strcmp(Type_Distribution,'uniform') == 1
            psi_x = 3/2*r - 1./(2*r);
            psi_0_pos = find(psi_x>=0, 1 );
            psi_0 = r(psi_0_pos);
            inv_psi_x = (r + sqrt(r.^2 + 3))/3;
            f_Vtype_radial = 1 ./ (3/2 + 1 ./ (2 * inv_psi_x.^2) );
            %plot(r,f_Vtype_radial)
            
        elseif (strcmp(Type_Distribution,'gaussian') == 1)||(strcmp(Type_Distribution,'beta') == 1)
            fx = f_type_radial / sum(f_type_radial*dx);
            
            % original virtual type cal.
            d = 2;
            Power = x .^ (d-1);
            mx = fx .* Power;
            Int = zeros(size(x));
            Int = (sum(mx) - cumsum(mx))*dx;
            psi_x = x - Int ./ mx;
            
            psi_0_pos = find(psi_x>=0, 1 );
            psi_0 = x(psi_0_pos);
            
            % original virtual type distribution cal.
            psi_diff = [diff(psi_x)/dx 2];
            
            x_hat = psi_x(psi_0_pos:end);
            inv_psi = x(psi_0_pos:end);
            
            f_invPsi_xHat = fx(psi_0_pos:end);
            psiDiff_invPsi_xHat = psi_diff(psi_0_pos:end);
            g_xHat = f_invPsi_xHat ./ psiDiff_invPsi_xHat;
            
            sum_prob = sum(diff(x_hat) .* g_xHat(2:end));
            
            % re-normalize fx
            % so that fx = 0 for x with negative virtual val.
            
            sum_positive = sum(fx(psi_0_pos:end))*dx;
            fx_normalized = fx;
            fx_normalized(1:psi_0_pos-1) = 0;
            fx_normalized(psi_0_pos:end) = fx(psi_0_pos:end) / (sum_positive);
            
            % normalized fx virtual type cal.
            Power = x .^ (d-1);
            mx_normalized = fx_normalized .* Power;
            Int_normalized = (sum(mx_normalized) - cumsum(mx_normalized))*dx;
            psi_x_normalized = x - Int_normalized ./ mx_normalized;
            
            psi_0_pos_normalized = find(psi_x_normalized>=0, 1 );
            
            % normalized virtual type distribution cal.
            psi_diff_normalized = [diff(psi_x_normalized)/dx 2];
            
            x_hat_normalized = psi_x_normalized(psi_0_pos_normalized:end);
            inv_psi_normalized = x(psi_0_pos_normalized:end);
            
            f_invPsi_xHat_normalized = fx_normalized(psi_0_pos_normalized:end);
            psiDiff_invPsi_xHat_normalized = psi_diff_normalized(psi_0_pos_normalized:end);
            g_xHat_normalized = f_invPsi_xHat_normalized ./ psiDiff_invPsi_xHat_normalized;
            
            sum_prob_normalized = sum(diff(x_hat_normalized) .* g_xHat_normalized(2:end));
            
            % Calculate quantization distortion
            
            % Resample g(xHat) function to be fr(x)
            fr_x = interp1([0 x_hat_normalized],[g_xHat_normalized(1) g_xHat_normalized],x);
            
            % re-normalize the distribution so that the integral on disk is 1
            S_x = 1/2^(d) *2*pi^(d/2)/gamma(d/2)*x.^(d-1); % area of d-1 dim sephere with radius 1 in positive orthant
            Prob_Int = sum(fr_x .* S_x) * dx;
            
            f_radius = fr_x / Prob_Int;
            f_Vtype_radial = f_radius;
            plot(r,f_Vtype_radial)
        end
        
        % convert f_Vtype_radial to grid
        f = zeros(1,Len);
        f_original = zeros(1,Len);
        Grid_pdf = 1./ zeros(length(x),length(y));
        for i = 1:1:Len
            rad = sqrt(X(i,1)^2 + X(i,2)^2);
            [temp,r_pos] = min(abs(r - rad));
            f(i) = f_Vtype_radial(r_pos);
            f_original(i) = f_type_radial(r_pos);
            Grid_pdf(X_pos(i,1),X_pos(i,2)) = f_Vtype_radial(r_pos);
        end
        figure(2)
        mesh(x_grid,y_grid,Grid_pdf)
        colormap hot
        title('Quantized Distribution')
        
    end
end


% Normalize the distribution
f_sum = sum(f);
f_int = f_sum * dx* dy;
f = f / f_int;
fprintf('done.\n');

%% Intial reconstruction point
fprintf('Generating initial partition...\n');
Pn = randperm(Len,N);
Pn = floor(Len/N:Len/N:Len);
Px = X(Pn,1);
Py = X(Pn,2);
P = [Px  Py];

% uniformly place initial point
% d_init = 1 / (sqrt(N)+1);
% Px = d_init:d_init:1 - d_init;
% Py = Px;
% P = [Px  Py];

%% Show the reconstruction point
if strcmp(Separability,'nonseparable') == 1
    figure(1)
    voronoi(Px,Py);
    hold on
    figure(1)
    plot(Px,Py,'*')
    
    hold off
    axis equal
    axis([0,1,0,1])
else
    figure(1)
    voronoi(Px,Py);
    hold on
    figure(1)
    plot(Px,Py,'*')
    
    hold on
    t = 0:dx:1;
    plot(t,sqrt(1 - t.^2),'b','Linewidth',1.5)
    hold off
    axis equal
    axis([0,1,0,1])
end
fprintf('done.\n');

%% Iteration

for iter = 1:1:Max_Iter
    fprintf('Iter: %d \n',iter)
    %% iter: find margins
    %D = 2*ones(length(x),length(x),length(Px)); % distance to each reconstruction point
    Quan = zeros(1,Len); % Denote the reconstruction point to each point
    
    for n = 1:1:Len
        x = X(n,1);
        y = X(n,2);
        d_min = 2;
        m = 0; % which point is the nearest
        for k = 1:1:length(Px)
            d = (x - P(k,1))^2 + (y - P(k,2))^2;
            if (d < d_min)
                m = k;
                d_min = d;
            end
        end
        Quan(n) = m;
    end
    
    %% iter: recompute reconstruction points
    Int = zeros(length(Px),2); % Integral of probability*position for each reconstruction point
    Sum = zeros(length(Px),1); % Sum of probability for each reconstruction point
    
    for n = 1:1:Len
        x = X(n,1);
        y = X(n,2);
        p = f(n);
        Int(Quan(n),1) = Int(Quan(n),1) + x*p;
        Int(Quan(n),2) = Int(Quan(n),2) + y*p;
        Sum(Quan(n)) = Sum(Quan(n)) + p;
    end
    
    P_new = zeros(size(P));
    for k = 1:1:length(Px)
        P_new(k,1) = Int(k,1) / Sum(k);
        P_new(k,2) = Int(k,2) / Sum(k);
    end
    
    P = P_new;
    
    if strcmp(Separability,'nonseparable') == 1
        figure(1)
        voronoi(P(:,1).',P(:,2).');
        hold on
        figure(1)
        plot(P(:,1).',P(:,2).','*')
        hold off
        axis equal
        axis([0,1,0,1])
    else
        figure(1)
        voronoi(P(:,1).',P(:,2).');
        hold on
        figure(1)
        plot(P(:,1).',P(:,2).','*')
        axis([0,1,0,1])
        hold on
        t = 0:dx:1;
        plot(t,sqrt(1 - t.^2),'b','Linewidth',1.5)
        hold off
        axis equal
        axis([0,1,0,1])
    end
    
    pause(0.05)
    
end


%% plot voronoi of type (if welfare) /virtual type (if revenue)
figure(1)
if strcmp(Separability,'nonseparable') == 1
    figure(1)
    voronoi(P(:,1).',P(:,2).');
    hold on
    figure(1)
    scatter(P(:,1).',P(:,2).','MarkerFaceColor','r','SizeData',11)
    hold off
    axis equal
    axis([0,1,0,1])
else
    figure(1)
    voronoi(P(:,1).',P(:,2).');
    [vx,vy] = voronoi(P(:,1).',P(:,2).');
    hold on
    figure(1)
    scatter(P(:,1).',P(:,2).','MarkerFaceColor','r','SizeData',11)
    axis([0,1,0,1])
    hold on
    t = 0:dx:1;
    plot(t,sqrt(1 - t.^2),'b','Linewidth',1.5)
    hold off
    axis equal
    axis([0,1,0,1])
end

% cutting lines out of the circle
if strcmp(Separability,'separable') == 1
    N_edge = length(vx); % vx, vy are end of the Voronoi edges.
    for i = 1:1:N_edge
        % i-th Voronoi edge
        dl = 0.005; % length segement of edges
        N_point_edge = ceil(sqrt((vx(1,i) - vx(2,i))^2 + (vy(1,i) - vy(2,i))^2)/dl) +1; % number of points that sketch the edge
        %N_point_edge = 5;
        x_edge = vx(1,i):(vx(2,i) - vx(1,i))/(N_point_edge-1) :vx(2,i);
        y_edge = vy(1,i):(vy(2,i) - vy(1,i))/(N_point_edge-1) :vy(2,i);
        for j = 1:1:N_point_edge
            if  x_edge(j) <0
                x_edge(j) = NaN;
                y_edge(j) = NaN;
            end
            if y_edge(j) <0
                x_edge(j) = NaN;
                y_edge(j) = NaN;
            end
            if x_edge(j)^2 + y_edge(j)^2 > 1
                x_edge(j) = NaN;
                y_edge(j) = NaN;
            end
        end
        
        figure(1)
        plot(x_edge,y_edge,'b-')
        axis([0,1,0,1])
        hold on
    end
    figure(1)
    scatter(P(:,1).',P(:,2).','MarkerFaceColor','r','SizeData',11)
    hold on
    t = 0:dx:1;
    plot(t,sqrt(1 - t.^2),'b','Linewidth',1.5)
    hold off
    axis equal
    axis([0,1,0,1])
end

if strcmp(Welfare_Revenue,'welfare') == 1
    FileName = ['Voronoi_' Type_Distribution '_' Welfare_Revenue '_' Separability '_n' num2str(N)]
    FigureTitle = ['Voronoi: ' Type_Distribution ', ' Welfare_Revenue ', n=' num2str(N)];
elseif strcmp(Welfare_Revenue,'revenue') == 1
    FileName = ['Voronoi_' Type_Distribution '_' Welfare_Revenue '_' Separability '_n' num2str(N) '_VirtualType'];
    FigureTitle = ['Voronoi (Virtual Type): ' Type_Distribution ', ' Welfare_Revenue ', n=' num2str(N) ];
end
figure(1)
title(FigureTitle)
xticks(0:0.2:1)
        yticks(0:0.2:1)

%% plot voronoi of original type (if revenue)
% figure(4)
% plot(psi_x)
% axis([0,length(psi_x),0,1])
if strcmp(Welfare_Revenue,'revenue') == 1
    
    P_original = zeros(size(P));
    for i = 1:1:N % plot the reconstruction point
        x_recon = P(i,1);
        y_recon = P(i,2);
        r_recon = sqrt(x_recon^2 + y_recon^2);
        [temp,r_original_pos] = min(abs(psi_x - r_recon));
        r_recon_original = r_original_pos * dx;
        x_recon_original = x_recon / r_recon * r_recon_original;
        y_recon_original = y_recon / r_recon * r_recon_original;
        P_original(i,1) = x_recon_original;
        P_original(i,2) = y_recon_original;
    end
    figure(3)
    scatter(P_original(:,1).',P_original(:,2).','MarkerFaceColor','r','SizeData',11)
    hold on
    
    
    N_edge = length(vx); % vx, vy are end of the Voronoi edges.
    for i = 1:1:N_edge
        % i-th Voronoi edge
        dl = 0.005; % length segement of edges
        N_point_edge = ceil(sqrt((vx(1,i) - vx(2,i))^2 + (vy(1,i) - vy(2,i))^2)/dl) +1; % number of points that sketch the edge
        %N_point_edge = 5;
        x_edge = vx(1,i):(vx(2,i) - vx(1,i))/(N_point_edge-1) :vx(2,i);
        y_edge = vy(1,i):(vy(2,i) - vy(1,i))/(N_point_edge-1) :vy(2,i);
        for j = 1:1:N_point_edge
            if  x_edge(j) <0
                x_edge(j) = NaN;
                y_edge(j) = NaN;
            end
            if y_edge(j) <0
                x_edge(j) = NaN;
                y_edge(j) = NaN;
            end
            if x_edge(j)^2 + y_edge(j)^2 > 1
                x_edge(j) = NaN;
                y_edge(j) = NaN;
            end
        end
        %
        r_edge = sqrt(x_edge.^2 + y_edge.^2);
        r_edge_original = zeros(size(x_edge));
        for j = 1:1:N_point_edge
            x_point = x_edge(j);
            y_point = y_edge(j);
            r_point = sqrt(x_point^2+y_point^2);
            [temp,r_original_pos] = min(abs(psi_x - r_point));
            r_edge_original(j) = r_original_pos * dx;
            %             x_point_original = x_point / r_point * r_original;
            %             y_point_original = y_point / r_point * r_original;
            %             x_edge_original(j) = x_point_original;
            %             y_edge_original(j) = y_point_original;
        end
        x_edge_original = x_edge ./ r_edge .* r_edge_original;
        y_edge_original = y_edge ./ r_edge .* r_edge_original;
        
        figure(3)
        plot(x_edge_original,y_edge_original,'b-')
        axis([0,1,0,1])
        hold on
    end
    t = 0:dx:1;
    figure(3)
    plot(t,sqrt(1 - t.^2),'b','Linewidth',1.5)
    axis equal
    axis([0,1,0,1])
    hold on
    
    
    t = 0:dx:psi_0;
    figure(3)
    plot(t,sqrt(psi_0^2 - t.^2),'b','Linewidth',1.5)
    axis equal
    axis([0,1,0,1])
    hold off
    
    FileName = ['Voronoi_' Type_Distribution '_' Welfare_Revenue '_' Separability '_n' num2str(N) '_OriginalType'];
    FigureTitle = ['Voronoi (Original Type): ' Type_Distribution ', ' Welfare_Revenue ', n=' num2str(N) ];
    figure(3)
    title(FigureTitle)
    xlabel('\bf{{\theta}_1}')
    ylabel('\bf{{\theta}_2}')
    xticks(0:0.2:1)
        yticks(0:0.2:1)
end

%% display chunk probability
%if strcmp(Welfare_Revenue,'welfare') == 1
Sum_quant = zeros(length(Px),1); % Sum of probability for each reconstruction point

for n = 1:1:Len
    x = X(n,1);
    y = X(n,2);
    p = f(n);
    Sum_quant(Quan(n)) = Sum_quant(Quan(n)) + p;
end

S_area = 1; % Here is S_area = 1 is beacuse the distribution has been normalized.


for i = 1:1:N
    figure(1)
    text(P(i,1)- label_offset_x,P(i,2)-label_offset_y,[num2str(Sum_quant(i)*dx*dy /S_area,2) ]);
    xlabel('\bf{{\theta}_1}')
    ylabel('\bf{{\theta}_2}')
    hold on
end
sum(Sum_quant)*dx*dy/S_area

% For revenue case
if strcmp(Welfare_Revenue,'revenue') == 1
    
    % Transform a new grid into the virtual space, remember the zone that
    % each point belong to, and sum
    
    Sum_quant = zeros(length(Px),1);
    
    for i = 1:1:Len % Spect at every point of a grid
        x_origin = X(i,1);
        y_origin = X(i,2);
        r_origin = sqrt(x_origin^2 + y_origin^2);
        r_origin_pos = round(r_origin / dx);
        %[temp,r_virtual_pos] = min(abs(psi_x - r_origin));
        %r_virtual = r_virtual_pos * dx;
        r_virtual = psi_x(r_origin_pos);
        if r_virtual <= 0
            continue;
        end
        x_virtual = x_origin * r_virtual/r_origin;
        y_virtual = y_origin * r_virtual/r_origin;
        
        d_min = 2;
        m = 0; % which point is the nearest
        for k = 1:1:length(Px)
            d = (x_virtual - P(k,1))^2 + (y_virtual - P(k,2))^2;
            if (d < d_min)
                m = k;
                d_min = d;
            end
        end
        Sum_quant(m) = Sum_quant(m) + f_type_radial(r_origin_pos);
        
    end
    
    Sum_quant = Sum_quant / sum(Sum_quant);
    
    P_original = P;
    
    for i = 1:1:N
        x_P = P(i,1);
        y_P = P(i,2);
        r_P = sqrt(x_P^2 + y_P^2);
        r_P_pos = round(r_P / dx);
        [temp,r_P_original_pos] = min(abs(psi_x - r_P));
        r_P_original = r_P_original_pos * dx;
        x_P_original = x_P * r_P_original / r_P;
        y_P_original = y_P * r_P_original / r_P;
        P_original(i,1) = x_P_original;
        P_original(i,2) = y_P_original;
    end
    
    for i = 1:1:N
        figure(3)
        text(P_original(i,1)-label_offset_x,P_original(i,2)-label_offset_y,[num2str(Sum_quant(i),2) ]);
        xlabel('\bf{{\theta}_1}')
        
        ylabel('\bf{{\theta}_2}')
        
        hold on
    end
    
end

