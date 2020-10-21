clear variables; close all; clc;
load('tops_probs.mat');

%% SMALL
disp('--- SMALL_SPARSE_NOCACHE ---');
topology = SMALL_SPARSE_NOCACHE.topology;
problem = SMALL_SPARSE_NOCACHE.problem;

DO_alt = 0;
DO_sub = 0;
N = 10;
for i=1:1:N
    [DO_best_alt, ~, ~, DO_best_sub, ~, ~] = mainAltSubFast(topology, problem);
    DO_alt = DO_alt + DO_best_alt;
    DO_sub = DO_sub + DO_best_sub;
end
DO_alt = DO_alt/N;
DO_sub = DO_sub/N;
DO_LRU = mainLRU(topology,problem);
fig1 = figure;
bar([DO_alt DO_sub DO_LRU]);
saveas(fig1,'SMALL_SPARSE_NOCACHE.fig');

disp('--- SMALL_SPARSE_SMALLCACHE ---');
topology = SMALL_SPARSE_SMALLCACHE.topology;
problem = SMALL_SPARSE_SMALLCACHE.problem;

DO_alt = 0;
DO_sub = 0;
N = 10;
for i=1:1:N
    [DO_best_alt, ~, ~, DO_best_sub, ~, ~] = mainAltSubFast(topology, problem);
    DO_alt = DO_alt + DO_best_alt;
    DO_sub = DO_sub + DO_best_sub;
end
DO_alt = DO_alt/N;
DO_sub = DO_sub/N;
DO_LRU = mainLRU(topology,problem);
fig2 = figure;
bar([DO_alt DO_sub DO_LRU]);
saveas(fig2,'SMALL_SPARSE_SMALLCACHE.fig');

disp('--- SMALL_DENSE_SMALLCACHE ---');
topology = SMALL_DENSE_SMALLCACHE.topology;
problem = SMALL_DENSE_SMALLCACHE.problem;

DO_alt = 0;
DO_sub = 0;
N = 10;
for i=1:1:N
    [DO_best_alt, ~, ~, DO_best_sub, ~, ~] = mainAltSubFast(topology, problem);
    DO_alt = DO_alt + DO_best_alt;
    DO_sub = DO_sub + DO_best_sub;
end
DO_alt = DO_alt/N;
DO_sub = DO_sub/N;
DO_LRU = mainLRU(topology,problem);
fig3 = figure;
bar([DO_alt DO_sub DO_LRU]);
saveas(fig3,'SMALL_DENSE_SMALLCACHE.fig');

%% MEDIUM
disp('--- MEDIUM_SPARSE_NOCACHE ---');
topology = MEDIUM_SPARSE_NOCACHE.topology;
problem = MEDIUM_SPARSE_NOCACHE.problem;

DO_alt = 0;
DO_sub = 0;
N = 10;
for i=1:1:N
    [DO_best_alt, ~, ~, DO_best_sub, ~, ~] = mainAltSubFast(topology, problem);
    DO_alt = DO_alt + DO_best_alt;
    DO_sub = DO_sub + DO_best_sub;
end
DO_alt = DO_alt/N;
DO_sub = DO_sub/N;
DO_LRU = mainLRU(topology,problem);
fig4 = figure;
bar([DO_alt DO_sub DO_LRU]);
saveas(fig4,'MEDIUM_SPARSE_NOCACHE.fig');

disp('--- MEDIUM_SPARSE_SMALLCACHE ---');
topology = MEDIUM_SPARSE_SMALLCACHE.topology;
problem = MEDIUM_SPARSE_SMALLCACHE.problem;

DO_alt = 0;
DO_sub = 0;
N = 10;
for i=1:1:N
    [DO_best_alt, ~, ~, DO_best_sub, ~, ~] = mainAltSubFast(topology, problem);
    DO_alt = DO_alt + DO_best_alt;
    DO_sub = DO_sub + DO_best_sub;
end
DO_alt = DO_alt/N;
DO_sub = DO_sub/N;
DO_LRU = mainLRU(topology,problem);
fig5 = figure;
bar([DO_alt DO_sub DO_LRU]);
saveas(fig5,'MEDIUM_SPARSE_SMALLCACHE.fig');

disp('--- MEDIUM_DENSE_SMALLCACHE ---');
topology = MEDIUM_DENSE_SMALLCACHE.topology;
problem = MEDIUM_DENSE_SMALLCACHE.problem;

DO_alt = 0;
DO_sub = 0;
N = 10;
for i=1:1:N
    [DO_best_alt, ~, ~, DO_best_sub, ~, ~] = mainAltSubFast(topology, problem);
    DO_alt = DO_alt + DO_best_alt;
    DO_sub = DO_sub + DO_best_sub;
end
DO_alt = DO_alt/N;
DO_sub = DO_sub/N;
DO_LRU = mainLRU(topology,problem);
fig6 = figure;
bar([DO_alt DO_sub DO_LRU]);
saveas(fig6,'MEDIUM_DENSE_SMALLCACHE.fig');

%% LARGE

disp('--- LARGE_SPARSE_NOCACHE ---');
topology = LARGE_SPARSE_NOCACHE.topology;
problem = LARGE_SPARSE_NOCACHE.problem;

DO_alt = 0;
DO_sub = 0;
N = 10;
for i=1:1:N
    [DO_best_alt, ~, ~, DO_best_sub, ~, ~] = mainAltSubFast(topology, problem);
    DO_alt = DO_alt + DO_best_alt;
    DO_sub = DO_sub + DO_best_sub;
end
DO_alt = DO_alt/N;
DO_sub = DO_sub/N;
DO_LRU = mainLRU(topology,problem);
fig7 = figure;
bar([DO_alt DO_sub DO_LRU]);
saveas(fig7,'LARGE_SPARSE_NOCACHE.fig');

disp('--- LARGE_SPARSE_SMALLCACHE ---');
topology = LARGE_SPARSE_SMALLCACHE.topology;
problem = LARGE_SPARSE_SMALLCACHE.problem;

DO_alt = 0;
DO_sub = 0;
N = 10;
for i=1:1:N
    [DO_best_alt, ~, ~, DO_best_sub, ~, ~] = mainAltSubFast(topology, problem);
    DO_alt = DO_alt + DO_best_alt;
    DO_sub = DO_sub + DO_best_sub;
end
DO_alt = DO_alt/N;
DO_sub = DO_sub/N;
DO_LRU = mainLRU(topology,problem);
fig8 = figure;
bar([DO_alt DO_sub DO_LRU]);
saveas(fig8,'LARGE_SPARSE_SMALLCACHE.fig');

