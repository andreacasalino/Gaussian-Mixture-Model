function print_with_colors2(points, labels)

N_cluster=size(labels,2);
colors=rand(3,N_cluster);

hold on;
for k=1:size(points, 1)
    col=colors*labels(k,:)';
    plot(points(k,1), points(k,2), '*', 'Color', col );
end
axis equal;