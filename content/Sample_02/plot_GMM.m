function plot_GMM(GMM, color)

colors=rand(length(GMM),3);

n=50;

x =linspace(-1,1,n);
y =linspace(-1,1,n);
[X,Y] = meshgrid(x, y);
Z=zeros(n,n);

for k=1:length(GMM)
    Z=add_cluster(GMM{k}, x,y,Z);
end

surf(X,Y,Z,'FaceColor', color);

end


function Z=add_cluster(cluster, x,y,Z)

inv_Cov=inv(cluster.Cov);
det_Cov=abs(det(cluster.Cov));
temp_den=sqrt((2*pi)^(length(cluster.Mean)) * det_Cov);

n=size(Z,1);

for r=1:n
    for c=1:n
        Z(r,c)= Z(r,c) + cluster.w * exp(-0.5 * ([x(r),y(c)]' - cluster.Mean)' * inv_Cov * ([x(r),y(c)]' - cluster.Mean) ) / temp_den;
    end
end

end