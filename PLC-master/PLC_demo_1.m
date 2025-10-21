% run_all_rhos_matlab.m
clear; clc;
rng('default');

% ========== 配置 ==========
% 假定 data (n x d) 已经在 workspace 中，target 也已加载
% 如果 target 是 n x 1 的类标，脚本会自动转换为 q x n 的 one-hot
% 如果 target 已是 q x n 的 one-hot，请保持原样。
if ~exist('data','var') || ~exist('target','var')
    error('请先加载 data 和 target 到 workspace（例如 load("lost.mat")）');
end

% 参数
rho_list = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40];  % 论文用于 constrained clustering 的集合
r_false = 2;     % 每个 partial-labeled 样本加入的假标签数（论文通过 r 控制）
num_repeats = 10; % 每个 rho 做多少个随机种子（论文通常是 10）
seeds = 0:(num_repeats-1);
outroot = 'results_matlab';
mkdir(outroot);

% 处理 target：将其规范为 q x n 的 one-hot 矩阵
if isvector(target) && length(target) == size(data,1)
    % target 是 n x 1 的标注（类别从 1..q）
    yvec = target(:);
    n = length(yvec);
    q = max(yvec);
    T = zeros(q, n);
    for i=1:n
        T(yvec(i), i) = 1;
    end
    target_onehot = T;
elseif size(target,1) > 1 && size(target,2) == size(data,1)
    % target 已是 q x n
    target_onehot = target;
    q = size(target_onehot,1);
    n = size(target_onehot,2);
else
    error('无法识别 target 形状，请确认是 n×1 或 q×n。');
end

% sanity
if size(data,1) ~= n
    error('data 行数应与 target 样本数一致');
end

% ========== 主循环：按 rho 与 seeds 生成采样、partial_target、运行 PLC ==========
summary_results = struct();

for ir = 1:length(rho_list)
    rho = rho_list(ir);
    fprintf('\n=== rho = %.3f ===\n', rho);
    results_acc = zeros(1, num_repeats);
    results_nmi = zeros(1, num_repeats);
    
    for s = 1:num_repeats
        seed = seeds(s);
        rng(seed);  % 固定随机种子以便复现
        
        % 1) 生成 train / test 索引（训练集 = partial-labeled）
        num_partial = max(1, round(rho * n));
        % ========== 可选：分层采样（分层保证每类都有代表） ==========
        stratified = true;
        if stratified
            % 按类分层抽取近似比例
            partial_idx = [];
            class_inds = cell(q,1);
            for c=1:q
                class_inds{c} = find(target_onehot(c,:) == 1);
            end
            for c=1:q
                nc = numel(class_inds{c});
                k = round(nc * rho);
                k = min(max(k,0), nc);
                if k > 0
                    choose = class_inds{c}(randperm(nc, k));
                    partial_idx = [partial_idx, choose];
                end
            end
            % 若总数不足 num_partial，则从剩余中随机补齐
            partial_idx = unique(partial_idx);
            if numel(partial_idx) < num_partial
                remain = setdiff(1:n, partial_idx);
                add = remain(randperm(numel(remain), num_partial - numel(partial_idx)));
                partial_idx = [partial_idx, add];
            elseif numel(partial_idx) > num_partial
                % 若超了，随机截断到 num_partial
                partial_idx = partial_idx(randperm(numel(partial_idx), num_partial));
            end
        else
            partial_idx = randperm(n, num_partial);
        end
        partial_idx = sort(partial_idx);
        train_idx = partial_idx(:);
        test_idx = setdiff(1:n, train_idx)';
        
        % 2) 生成 partial_target (q x n)
        Y_partial = zeros(q, n);  % 默认 0
        % 对训练集：把真实标签置1，并加 r_false 个错误标签
        for ii = 1:length(train_idx)
            idx = train_idx(ii);
            true_c = find(target_onehot(:, idx), 1);
            Y_partial(true_c, idx) = 1;
            % 候选假标签池
            pool = setdiff(1:q, true_c);
            if ~isempty(pool)
                m = min(r_false, numel(pool));
                if m > 0
                    fake = pool(randperm(numel(pool), m));
                    Y_partial(fake, idx) = 1;
                end
            end
        end
        % 对测试集：按照论文，测试样本的候选集设为全 1（即 y_test = 1_q）
        for ii = 1:length(test_idx)
            idx = test_idx(ii);
            Y_partial(:, idx) = 1;  % 表示测试时候选集合为全标签（paper 中的设定）
        end
        
        % 3) 保存采样索引与 partial_target 以便复现
        outdir = fullfile(outroot, sprintf('rho_%.3f_seed_%02d', rho, seed));
        if ~exist(outdir, 'dir'); mkdir(outdir); end
        save(fullfile(outdir, 'train_idx.mat'), 'train_idx');
        save(fullfile(outdir, 'test_idx.mat'), 'test_idx');
        save(fullfile(outdir, 'Y_partial.mat'), 'Y_partial');
        save(fullfile(outdir, 'config.mat'), 'rho', 'r_false', 'seed');
        
        % 4) 运行 PLC（使用你现有的 demo 代码结构）
        % 你的原 demo 是：
        % train_data = data(train_idx{it},:); train_p_target = partial_target(:,train_idx{it});
        % groups = PLC(train_data,train_p_target,test_data,para);
        %
        % 这里直接调用 PLC（注意 PLC 接口：train_p_target 为 q x n_train）
        train_data = data(train_idx, :);
        train_data = zscore(train_data);
        test_data = data(test_idx, :);
        test_data = zscore(test_data);
        train_p_target = Y_partial(:, train_idx);   % q x n_train
        test_target_mat = target_onehot(:, test_idx); % q x n_test for evaluation
        
        % ========== 确保 PLC 的接口一致 ==========
        % 若 PLC 的签名与你示例不同（比如需要 train_p_target 转置等），请按你的实现调整
        groups = PLC(train_data, train_p_target, test_data, para);  % groups: 1 x n_test 或 n_test x 1
        
        % 5) 评估（确保 CalMetrics 接受 target_onehot 的列向量形式）
        % 如果你的 CalMetrics 接受标签向量 (n_test x 1) 的形式，需将 one-hot 转回类标
        if size(test_target_mat,1) > 1
            [~, gt_labels] = max(test_target_mat, [], 1);
        else
            gt_labels = test_target_mat; % 原本就是类标
        end
        if size(groups,1) > size(groups,2)
            groups = groups'; % 保证为行向量
        end
        [acc, nmi] = CalMetrics(gt_labels(:), groups(:));
        
        results_acc(s) = acc;
        results_nmi(s) = nmi;
        
        fprintf('rho=%.3f seed=%02d -> ACC=%.4f, NMI=%.4f\n', rho, seed, acc, nmi);
        save(fullfile(outdir, 'metrics.mat'), 'acc', 'nmi');
    end
    
    % 汇总当前 rho 的结果并保存
    summary_results(ir).rho = rho;
    summary_results(ir).accs = results_acc;
    summary_results(ir).nmis = results_nmi;
    summary_results(ir).mean_acc = mean(results_acc);
    summary_results(ir).std_acc = std(results_acc);
    summary_results(ir).mean_nmi = mean(results_nmi);
    summary_results(ir).std_nmi = std(results_nmi);
    
    fprintf('>>> rho=%.3f summary => ACC mean=%.4f std=%.4f | NMI mean=%.4f std=%.4f\n', ...
        rho, mean(results_acc), std(results_acc), mean(results_nmi), std(results_nmi));
    
    save(fullfile(outroot, sprintf('summary_rho_%.3f.mat', rho)), 'results_acc', 'results_nmi', '-v7.3');
end

% 全部 rho 汇总
save(fullfile(outroot, 'summary_all_rhos.mat'), 'summary_results', '-v7.3');
disp('All done.');
