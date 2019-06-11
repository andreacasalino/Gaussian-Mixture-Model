%% GMM demo

clear; clc;
close all;

%compare the pdf of the model, and the one leart from samples
GMM_rand=import_GMM('GMM_random');
GMM_learnt=import_GMM('GMM_trained');

figure;

subplot(1,2,1);
hold on;
title('original model');
plot_GMM(GMM_rand, 'r');
view([50 64]);

subplot(1,2,2);
hold on;
title('learnt from samples retrieved from the original');
plot_GMM(GMM_learnt, 'b');
view([50 64]);




%plot the classification made about the points sampled
P_GMM=load('__Sampled_points');
P_GMM_labels=load('__Sampled_points_classification');

figure;
print_with_colors2(P_GMM , P_GMM_labels);