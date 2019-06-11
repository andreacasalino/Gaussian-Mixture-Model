%% K_means demo

clear; clc;
close all;


P=load('__Sampled_points');

%plot the entire set of samples
figure;
hold on;
plot(P(:,1), P(:,2),'*');
axis equal;

%plot the result of the k means: all the points
%with the same color are in the same cluster
figure;
print_with_colors(load('__Clustered_points'));
