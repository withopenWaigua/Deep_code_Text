
load data1.mat  %这是网上关于消费的一个案例数据，有31个样本和8个指标，详情看文件夹内excel数据。

[n,p] = size(x);  % n是样本个数，p是指标个数
disp('样本个数，指标个数：')
disp([n,p])
%% 第一步：对数据x标准化为X
X=zscore(x);   % matlab内置的标准化函数（x-mean(x)）/std(x)   z标准化
disp('标准化后的矩形是：')    %使数据合理，各个数值的影响相似，而不会过大
disp(X)    
%% 第二步：计算样本相关系数矩阵
R = cov(X);
disp('样本相关系数矩阵为：')
disp(R)

%% 第三步：计算特征值和特征向量

[V,D] = eig(R);  % V是对应的特征向量矩阵  D是特征值构成的对角矩阵

%% 第四步：计算主成分贡献率和累计贡献率
lambda = diag(D);  % diag函数用于得到一个矩阵的主对角线元素值(返回的是列向量)
lambda = lambda(end:-1:1);  % 因为lambda向量是从小大到排序的，我们将其调个头
contribution_rate = lambda / sum(lambda);  % 计算贡献率
cum_contribution_rate = cumsum(lambda)/ sum(lambda);   % 计算累计贡献率  cumsum是求累加值的函数 sum是指标数量（列）
disp('特征值为：')
disp(lambda')  % 转置为行向量，方便展示
disp('贡献率为：')
disp(contribution_rate')% 转置为行向量，方便展示
disp('累计贡献率为：')
disp(cum_contribution_rate')% 转置为行向量，方便展示
disp('与特征值对应的特征向量矩阵为：')
V=rot90(V)'; %由于之前把特征值反转了，因此也要把对应的特征向量矩阵反转
disp(V)


%% 提取主成分处理后的值
need_num =input('请输入需要保存的主成分的个数:  ');
F = zeros(n,need_num);  %初始化保存主成分的矩阵（每一列是一个主成分）
for i = 1:need_num
    ai = V(:,i)';   % 将第i个特征向量取出，并转置为行向量
    Ai = repmat(ai,n,1);   % 将这个行向量重复n次，构成一个n*p的矩阵
    F(:, i) = sum(Ai .* X, 2);  % 对标准化的数据求了权重后要计算每一行的和
end