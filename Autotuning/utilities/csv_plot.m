
%a = importdata('matlab_test_log.csv');
a = importdata('~/Code/thesis/Autotuning/examples/laplace3d/results/laplace3d.csv');

cols = a.colheaders;
data = a.data;

nscore = 0;

for n=1:length(cols)
  if (cols{n}(1:6)==' Score')
    nscore = nscore+1;
  end
end

nvar = length(cols) - nscore - 1;

Y = data(:,end);
bar(Y)
xlabel('test')
ylabel('time')

savefig('~/Code/thesis/Autotuning/examples/laplace3d/results/laplace3d.fig')