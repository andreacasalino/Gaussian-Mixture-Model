function GMM=import_GMM(file)

data=load(file);

n=size(data,2);
n=(n-2);

N_cluster=size(data,1) / n;

line=0;
for cl=1:N_cluster
    GMM{cl}.w        = data(line+1, 1);
    GMM{cl}.Mean = data((line +1) : (line +n) , 2);
    GMM{cl}.Cov     = data((line +1) : (line +n) , 3:end);
    
    line=line+n;
end