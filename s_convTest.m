% Testing subgradient projection method on a simple topology
%      o -> MC
%      |
%      * -> SC2
%      |
%      * -> SC1
%     / \
% U1 .   . U2

clear all; close all; clc;

noise = 1; % Noise power
lambda1 = 1; % Request rate parameter for U1
lambda2 = 1.5; % Request rate parameter for U2

syms s1 s2 s3 s4 % s1: SC1-U1 tx power, s2: SC1-U2 tx power, s3: SC2-SC1 tx power, s4: MC-SC2 tx power
syms y_sc1_1 y_sc1_2 y_sc2_1 y_sc2_2 % y_sc1: Caching variable at SC for file 1, y_sc2: Caching variable at SC for file 2

x1 = s1/(noise + s2 + s4/81); % SINR for U1
x2 = s2/(noise + s1 + s4/81); % SINR for U2
x3 = s3/(noise + s4/16);      % SINR for SC1
x4 = s4/(noise + s1 + s2);    % SINR for SC2

f1 = 1/log2(1+x1);            % Delay at SC1-U1 hop
f2 = 1/log2(1+x2);            % Delay at SC1-U2 hop
f3 = 1/log2(1+x3);            % Delay at SC2-SC1 hop
f4 = 1/log2(1+x4);            % Delay at MC-SC2 hop

g1_1 = 1 - piecewise(y_sc1_1<1, y_sc1_1, 1); % Relaxed caching sum for U1's request at SC1
g1_2 = 1 - piecewise(y_sc1_2<1, y_sc1_2, 1); % Relaxed caching sum for U2's request at SC1
g2_1 = 1 - piecewise(y_sc1_1+y_sc2_1<1, y_sc1_1+y_sc2_1, 1); % Relaxed caching sum for U1's request at SC2
g2_2 = 1 - piecewise(y_sc1_2+y_sc2_2<1, y_sc1_2+y_sc2_2, 1); % Relaxed caching sum for U2's request at SC2

D = lambda1*(f1+f3*g1_1+f4*g2_1) + lambda2*(f2+f3*g1_2+f4*g2_2); % Objective function

%grad = jacobian(D,[s1; s2; s3; s4; y_sc1_1; y_sc1_2; y_sc2_1; y_sc2_2]);
% s = [10 10 10 10];
%y = [0.6 0.4 0.4 0.6];
% n_grad = double(subs(grad,[s1 s2 s3 s4 y_sc1_1 y_sc1_2 y_sc2_1 y_sc2_2], [s y]));

s = [1 1 1 1];
y = 0.01:0.01:1;

n_D = double(subs(D,{s1, s2, s3, s4, y_sc1_1, y_sc1_2, y_sc2_1, y_sc2_2},{s(1), s(2), s(3), s(4), y, 1-y, 0.4, 0.6}));

figure;
plot(y, n_D);
hold on;
plot(1-y, n_D);