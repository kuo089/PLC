% PLC_demo_MSRCV2.m
% 依 MSRCv2_Sample.mat 的字段(data/partial_target/target/train_idx/test_idx)
% 严格按原论文示例：每折用 train 的部分标签训练，只在 test 上评测

clear; clc;

% -------------------- 加载数据 --------------------
load('MSRCv2_Sample.mat');  % 提供 data, partial_target (C x N), target (C x N), train_idx/test_idx (10x1 cell)

% -------------------- 超参数（按你给的原始设置） --------------------
para.alpha   = 0.01;
para.beta    = 0.01;
para.gamma   = 10;
para.lambda  = 1;
para.mu      = 1;
para.maxiter = 10;
para.maxit   = 5;
para.k       = 10;

ACC = [];
NMI = [];

num_folds = numel(train_idx);
fprintf('MSRCv2_Sample | folds=%d | k=%d | alpha=%.3g beta=%.3g gamma=%.3g | lambda=%.1f mu=%.1f | maxiter=%d maxit=%d\n', ...
    num_folds, para.k, para.alpha, para.beta, para.gamma, para.lambda, para.mu, para.maxiter, para.maxit);

for it = 1:num_folds
    % -------------------- 划分与标准化 --------------------
    tr = train_idx{it}(:);
    te = test_idx{it}(:);

    train_data = data(tr, :);
    test_data  = data(te, :);

    % 按原始代码风格做 zscore（各自独立标准化）
    train_data = zscore(train_data);
    test_data  = zscore(test_data);

    % 训练端部分标签（C x |train|），测试端真值（C x |test|）
    train_p_target = partial_target(:, tr);  % 注意：保持为 (classes x samples)
    test_target    = target(:, te);          % 同上

    % -------------------- 调用主流程 --------------------
    % 原论文代码接口：PLC(train_data, train_p_target, test_data, para)
    groups = PLC(train_data, train_p_target, test_data, para);  % 期望返回 |test| x 1 的簇标签(或1..C类标)

    % -------------------- 评测（只在 test 上） --------------------
    [acc, nmi] = CalMetrics(test_target, groups);

    ACC = [ACC, acc];
    NMI = [NMI, nmi];

    fprintf('Fold %2d/%d  ACC=%.4f  NMI=%.4f\n', it, num_folds, acc, nmi);
end

fprintf('\n=== Final on MSRCv2_Sample ===\n');
fprintf('ACC mean±std: %.4f ± %.4f\n', mean(ACC), std(ACC));
fprintf('NMI mean±std: %.4f ± %.4f\n', mean(NMI), std(NMI));
