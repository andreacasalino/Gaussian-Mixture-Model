%% uni-variate

clear; clc;
close all;

x=linspace(-4,4,500);

figure;
hold on;
plot(x,normpdf(x),'r','linewidth', 2);
legend('\phi_{(0,1)}(x)','interpreter', 'latex');
set(gca,'FontSize', 25);

figure; 
hold on;
F=[];
for k=[2,5,10]
    F=[F;normpdf(x,0,sqrt(k))];
end
F=[F;normpdf(x, 2, 1)];
plot(x, F','linewidth', 2);
legend({'$\phi_{(0,2)}(x)$','$\phi_{(0,5)}(x)$','$\phi_{(0,10)}(x)$','$\phi_{(2,1)}(x)$'},'interpreter', 'latex');
set(gca,'FontSize', 25);

%% GMM uni-variate

clear; clc;
close all;

x=linspace(-4,4,500);

means = [1.9852   -0.3957   -3.3294]; % rand(1,3)*8 - 4;
vars = [0.8131    1.2400    1.0429]; %abs(rand(1,3)*3 - 1.5);
w = [0.3, 0.6, 0.1];

pdf = normpdf(x, means(1), vars(1)) * (1);
for k=2:length(w)
    pdf = pdf + normpdf(x, means(k), vars(k)) * (k);
end

figure;
hold on;
plot(x,pdf,'r','linewidth', 2);
F= [];
for k=1:length(means)
    F= [F;normpdf(x,means(k),vars(k))];
    labels{k+1} = ['$\phi_{(\mu_', num2str(k),' \Sigma_' , num2str(k), ')}(x)$'];
end
labels{1} = '$f_{GMM}(x) = \lambda_1 \phi_{(\mu_1, \Sigma_1)}(x) + \lambda_2 \phi_{(\mu_2, \Sigma_2)}(x) + \lambda_3 \phi_{(\mu_3, \Sigma_3)}(x)$';
plot(x,F','--','linewidth', 1);
legend(labels,'interpreter', 'latex');
set(gca,'FontSize', 25);

%% multi variate

clear; clc;
close all;

Mu = [2,2]';
sigma = diag([3,1]);
Sigma = sigma;

teta = 30*pi/180;
R= [cos(teta), -sin(teta);
       sin(teta), cos(teta)];
Sigma = R*Sigma*R';

[X,Y] = meshgrid(linspace(-2,5,50));  

Z=X;
for x=1:length(X)
    for y=1:length(Y)
        Z(x,y) = mvnpdf([X(x,y) , Y(x,y)]',Mu,Sigma);
    end
end

figure;
hold on;
hs = surf(X,Y,Z+0.03);
hs.FaceColor = [1,0,0];
hs.FaceAlpha = 1;
hs.EdgeColor = [0,0,0];
hc = contour(X,Y,Z);
xlabel('$x_1$','interpreter','latex');
ylabel('$x_2$','interpreter','latex');
zlabel('$\Phi(X)$','interpreter','latex');
set(gca,'FontSize', 25);

L{1}=[Mu - R(:,1)*4, Mu + R(:,1)*4];
L{2}=[Mu - R(:,2)*4, Mu + R(:,2)*4];

plot3(L{1}(1,:),L{1}(2,:), zeros(1,2), 'k--');
plot3(L{2}(1,:),L{2}(2,:), zeros(1,2), 'k--');
figure;
hold on;
contour(X,Y,Z);
plot3(L{1}(1,:),L{1}(2,:), zeros(1,2), 'k--');
plot3(L{2}(1,:),L{2}(2,:), zeros(1,2), 'k--');
xlabel('$x_1$','interpreter','latex');
ylabel('$x_2$','interpreter','latex');
set(gca,'FontSize', 25);
