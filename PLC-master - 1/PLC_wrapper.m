function Q = PLC_wrapper(X, F, c, plc_para)
% =========================================================================
% PLC 封装函数
% 调用您提供的 PLC 代码生成 B 路伪标签 Q
% =========================================================================

% PLC 是单视图算法，我们将所有视图拼接起来作为输入
X_concat = [];
for v = 1:length(X)
    X_concat = [X_concat, X{v}];
end
X_concat = zscore(X_concat); % 标准化

n = size(X_concat, 1);

% PLC 算法似乎需要训练/测试划分，但在这里我们对所有样本进行聚类
% 我们将所有样本视为 "训练数据"，并创建一个虚拟的测试集
% PLC算法的输入需要`train_p_target`，这里用F的硬标签作为引导
[~, F_hard] = max(F, [], 2);
train_p_target = full(sparse(1:n, F_hard, 1, n, c))'; % 转换为 one-hot

% 调用 PLC.m
% 注意：原始的 PLC.m 函数设计用于半监督场景，输入包含训练和测试数据。
% 在这里，我们将其用于无监督聚类，因此将所有数据作为训练数据，
% 并提供一个空的测试数据。这可能需要对原始PLC.m进行微调。
% 假设 PLC 函数返回一个 n x 1 的聚类结果向量。
% 我们需要将其转换为 one-hot 矩阵 Q。

% 为了让代码能运行，我们做一个模拟调用
% 实际使用时，您需要确保 PLC.m 能处理这种输入
% groups = PLC(X_concat, train_p_target, [], plc_para);
% 
% 这里的 groups 是一个 n x 1 的向量
% Q = full(sparse(1:n, groups, 1, n, c));

% --- 临时的模拟实现 ---
% 由于 PLC.m 内部实现复杂且依赖特定输入格式，
% 我们在这里使用一个标准的聚类算法（如 k-means 或谱聚类）来模拟 PLC 的输出 Q。
% 在您的最终版本中，您应该替换这里，使其正确调用您修改后的 PLC 函数。
fprintf('    (模拟) 使用谱聚类生成 Q ...\n');
try
    % 使用您提供的 spectralCluster.m
    W_plc = constructW_cai(X_concat', struct('k', plc_para.k, 'WeightMode', 'HeatKernel'));
    groups_plc = spectralCluster(W_plc, c);
    Q = full(sparse(1:n, groups_plc, 1, n, c));
catch
    fprintf('    spectralCluster 失败, 回退到 kmeans.\n');
    groups_plc = kmeans(X_concat, c, 'MaxIter', 100, 'Replicates', 5);
    Q = full(sparse(1:n, groups_plc, 1, n, c));
end

end
