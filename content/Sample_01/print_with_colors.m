function print_with_colors(points)

N_cluster=max(points(:,1))+1;
colors=rand(N_cluster,3);

hold on;
for k=1:size(points, 1)
    plot(points(k,2), points(k,3), '*', 'Color', colors(points(k,1) + 1 ,:) );
end
axis equal;