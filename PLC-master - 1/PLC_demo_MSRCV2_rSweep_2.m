% PLC_demo_MSRCV2_rSweep_fixed.m
% 一次性扫 rho ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.40}
% 依赖：Optimization Toolbox、Statistics and Machine Learning Toolbox

clear; clc;
rng('default'); % 固定随机种子

% -------------------- 加载数据 --------------------
% 注意：这里只加载data和target，因为我们将重新生成部分标签和训练/测试划分
load('MSRCv2_Sample.mat', 'data', 'target');
[N, d] = size(data);
[C, ~] = size(target);

% -------------------- 超参数 --------------------
para.alpha   = 0.01;
para.beta    = 0.01;
para.gamma   = 10;
para.lambda  = 1;
para.mu      = 1;
para.maxiter = 10;
para.maxit   = 5;
para.k       = 10;

% 要扫描的 rho，对应论文中 p (Proportion of partial labeled samples)
rho_list = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40];
r_false = 2; % 每个 partial-labeled 样本加入的假标签数 (对应论文中的 r)
num_repeats = 10; % 每个 rho 重复运行的次数

% 结果存放
ACC_tbl = zeros(numel(rho_list), 2); % [mean, std]
NMI_tbl = zeros(numel(rho_list), 2); % [mean, std]

fprintf('MSRCv2_Sample | r_false=%d | k=%d | alpha=%.3g beta=%.3g gamma=%.3g | lambda=%.1f mu=%.1f | maxiter=%d maxit=%d\n\n', ...
    r_false, para.k, para.alpha, para.beta, para.gamma, para.lambda, para.mu, para.maxiter, para.maxit);

% -------------------- 主循环：按 rho 与 seeds 运行 --------------------
for r = 1:numel(rho_list)
    rho = rho_list(r);
    fprintf('\n=== rho = %.3f ===\n', rho);
    
    ACC = zeros(1, num_repeats);
    NMI = zeros(1, num_repeats);
    
    for s = 1:num_repeats
        % 1) 固定随机种子以便复现
        rng(s, 'twister');

        % 2) 生成训练/测试集索引（按 rho 比例抽取）
        num_partial = max(1, round(rho * N));
        
        % 分层采样以保证每类都有代表
        gt_labels_vec = max(target, [], 1)';
        cv = cvpartition(gt_labels_vec, 'KFold', round(1/rho));
        train_idx = find(training(cv, 1));
        test_idx = find(test(cv, 1));
        
        % 3) 生成部分标签
        partial_target = zeros(C, N);
        
        % 对训练集中的每个样本，保留真标签并加入 r_false 个假标签
        for ii = 1:numel(train_idx)
            idx = train_idx(ii);
            true_c = find(target(:, idx), 1);
            partial_target(true_c, idx) = 1;
            
            pool = setdiff(1:C, true_c);
            if ~isempty(pool)
                k = min(r_false, numel(pool));
                if k > 0
                    fake_labels = pool(randperm(numel(pool), k));
                    partial_target(fake_labels, idx) = 1;
                end
            end
        end
        
        % 对测试集中的样本，按照论文约定，候选标签集设为全 1
        for ii = 1:numel(test_idx)
            idx = test_idx(ii);
            partial_target(:, idx) = 1;
        end
        
        % 4) 划分数据并标准化（各自独立 zscore）
        train_data = zscore(data(train_idx, :));
        test_data = zscore(data(test_idx, :));
        
        % 5) 提取本轮训练用的部分标签和测试真值
        train_p_target = partial_target(:, train_idx);
        test_target_te = target(:, test_idx);
        
        % 6) 调用主流程
        groups = PLC(train_data, train_p_target, test_data, para);
        
        % 7) 评测
        [acc, nmi] = CalMetrics(test_target_te, groups);
        
        ACC(s) = acc;
        NMI(s) = nmi;
        
        fprintf('  Seed %2d/%d  ACC=%.4f  NMI=%.4f\n', s, num_repeats, acc, nmi);
    end
    
    ACC_tbl(r, :) = [mean(ACC), std(ACC)];
    NMI_tbl(r, :) = [mean(NMI), std(NMI)];
    
    fprintf('--- rho=%.2f  ACC mean±std: %.4f ± %.4f | NMI mean±std: %.4f ± %.4f\n\n', ...
        rho, ACC_tbl(r,1), ACC_tbl(r,2), NMI_tbl(r,1), NMI_tbl(r,2));
end

% -------------------- 总结输出 --------------------
fprintf('\n==== Summary over rho ====\n');
for r = 1:numel(rho_list)
    fprintf('rho=%.2f | ACC: %.4f ± %.4f | NMI: %.4f ± %.4f\n', ...
        rho_list(r), ACC_tbl(r,1), ACC_tbl(r,2), NMI_tbl(r,1), NMI_tbl(r,2));
end