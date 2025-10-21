function pred_labels = PLKNN_predict(X_train, Y_train, X_test, k)
% =========================================================================
% PLKNN_predict: 基于K近邻的偏标记学习预测函数
%
% 描述:
%   对于每一个测试样本，在训练集中找到其K个最近邻，然后通过在这些
%   邻居的候选标签集中投票，来决定该测试样本的最终标签。
%
% 输入:
%   X_train   - 训练集特征 (cell array, 每个cell是一个视图)
%   Y_train   - 训练集偏标记矩阵 (n_train x c)
%   X_test    - 测试集特征 (cell array)
%   k         - K-近邻的数量
%
% 输出:
%   pred_labels - 对测试集样本的预测标签 (n_test x 1)
% =========================================================================

V = length(X_train);
[n_train, c] = size(Y_train);
n_test = size(X_test{1}, 1);

% --- 特征拼接 ---
% 将多视图数据拼接成一个大的单视图矩阵，以便计算距离
X_train_cat = [];
X_test_cat = [];
for v = 1:V
    X_train_cat = [X_train_cat, X_train{v}];
    X_test_cat = [X_test_cat, X_test{v}];
end

% --- 对每个测试样本进行预测 ---
pred_labels = zeros(n_test, 1);
fprintf('  PL-KNN 正在预测 %d 个测试样本...\n', n_test);
for i = 1:n_test
    test_sample = X_test_cat(i, :);
    
    % 计算当前测试样本与所有训练样本的欧氏距离
    distances = sqrt(sum((X_train_cat - test_sample).^2, 2));
    
    % 找到最近的 k 个邻居的索引
    [~, sorted_indices] = sort(distances, 'ascend');
    neighbor_indices = sorted_indices(1:k);
    
    % --- 投票过程 ---
    % 收集所有邻居的候选标签
    neighbor_candidate_labels = Y_train(neighbor_indices, :);
    
    % 计算每个类别的得票数
    % votes(j) 表示类别 j 获得了多少票
    votes = zeros(1, c);
    for j = 1:c
        votes(j) = sum(neighbor_candidate_labels(:, j));
    end
    
    % 选出得票最多的类别作为预测结果
    [~, predicted_class] = max(votes);
    
    % 处理平票情况：如果多个类别得票数相同，则选择索引最小的那个
    pred_labels(i) = predicted_class(1); 
end

end

