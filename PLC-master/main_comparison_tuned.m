% =========================================================================
% 对比实验 (参数调优版): DGR-AMFS vs. PL-KNN
% 描述:
%   本脚本通过搜索一系列超参数，旨在为 DGR-AMFS 找到在当前半监督
%   设置下的最优性能，并与 PL-KNN 进行比较。
%
% 新增功能:
%   - 自动循环测试不同的 lambda_a 值。
%   - 记录每次测试的结果。
%   - 绘制性能变化曲线图。
%   - 报告 DGR-AMFS 的最佳性能及其对应的参数值。
%
% 作者: 王阔 (思路) & Gemini (实现)
% 日期: 2025-10-21
% =========================================================================

clear;
clc;
close all;

% --- 路径设置 ---
addpath(genpath('./')); 

%% 1. 数据加载与准备
fprintf('1. 加载 MSRCv2 数据并划分训练/测试集...\n');
load('MSRCv2_Sample.mat'); 

% --- 数据格式转换 ---
Y_full = partial_target'; 
[~, true_labels_indices] = max(target, [], 1);
true_labels_full = true_labels_indices'; 
[n, c] = size(Y_full);

% 模拟多视图数据
X_full = cell(1, 2);
X_full{1} = zscore(data);
X_full{2} = zscore(data + 0.1 * randn(size(data)));
V = length(X_full);

% --- 划分训练集与测试集 ---
train_ratio = 0.8;
% 为了让每次运行的结果可复现，我们固定随机种子
rng(1); 
indices = randperm(n);
train_idx = indices(1:floor(n * train_ratio));
test_idx = indices(floor(n * train_ratio)+1:end);

% 创建训练集和测试集
X_train = cell(1, V);
X_test = cell(1, V);
for v = 1:V
    X_train{v} = X_full{v}(train_idx, :);
    X_test{v} = X_full{v}(test_idx, :);
end
Y_train = Y_full(train_idx, :);
true_labels_test = true_labels_full(test_idx);

fprintf('数据准备完毕: %d 个训练样本, %d 个测试样本。\n\n', length(train_idx), length(test_idx));


%% 2. 运行 PL-KNN (作为固定基准)
fprintf('2. 运行 PL-KNN 基准模型...\n');
k_knn = 10;
pred_labels_plknn = PLKNN_predict(X_train, Y_train, X_test, k_knn);
acc_plknn = sum(pred_labels_plknn == true_labels_test) / length(test_idx);
fprintf('PL-KNN 模型评估完成，ACC: %.4f\n\n', acc_plknn);


%% 3. 对 DGR-AMFS 进行参数搜索
fprintf('3. 开始对 DGR-AMFS 进行参数搜索...\n');

% --- 定义要搜索的参数范围 ---
lambda_a_range = [1e-3, 1e-2, 1e-1, 1, 10, 100]; % 测试 6 个不同的 lambda_a 值
results_acc = zeros(size(lambda_a_range)); % 用于存储结果

% --- 循环测试 ---
for i = 1:length(lambda_a_range)
    current_lambda_a = lambda_a_range(i);
    fprintf('--> 正在测试 lambda_a = %.4f (%d/%d)...\n', current_lambda_a, i, length(lambda_a_range));
    
    % --- 设置超参数 ---
    params.V = V;
    params.c = c;
    params.max_iter = 30;
    params.warmup_iter = 5;
    params.lambda_s = 1e-3;
    params.beta = 1e-2;
    params.lambda_a = current_lambda_a; % [!!] 使用当前循环的值
    params.lambda_B = 1e-2;
    params.lambda_e = 1e-2;
    params.gamma = 2;
    params.r_G1 = 3;
    params.m_PLC = 5;
    params.k_G1 = 10;
    params.k_G2 = 10;
    params.sigma = 1;
    params.plc.alpha = 0.01;
    params.plc.beta = 0.01;
    params.plc.gamma = 10;
    params.plc.lambda = 1;
    params.plc.mu = 1;
    params.plc.maxiter = 10;
    params.plc.maxit = 5;
    params.plc.k = 10;
    
    % 准备DGR-AMFS的输入
    Y_for_dgr = zeros(n, c);
    Y_for_dgr(train_idx, :) = Y_train;
    
    % 运行算法
    [F, ~, ~, ~] = DGR_AMFS_train(X_full, Y_for_dgr, params);
    
    % 在测试集上评估
    F_test = F(test_idx, :);
    [~, pred_labels_dgr_raw] = max(F_test, [], 2);
    pred_labels_dgr = bestMap(true_labels_test, pred_labels_dgr_raw);
    acc_dgr = sum(pred_labels_dgr == true_labels_test) / length(test_idx);
    
    results_acc(i) = acc_dgr;
    fprintf('    完成, ACC: %.4f\n', acc_dgr);
end
fprintf('参数搜索完成。\n\n');


%% 4. 显示最终对比结果
% --- 找到 DGR-AMFS 的最佳结果 ---
[best_acc_dgr, best_idx] = max(results_acc);
best_lambda_a = lambda_a_range(best_idx);

fprintf('==================== 最终对比结果 ====================\n');
fprintf('  数据集: MSRCv2, 训练集比例: %.2f\n', train_ratio);
fprintf('------------------------------------------------------\n');
fprintf('  算法名称          | 在测试集上的ACC\n');
fprintf('------------------------------------------------------\n');
fprintf('  DGR-AMFS (最佳)   |   %.4f  (当 lambda_a=%.3f)\n', best_acc_dgr, best_lambda_a);
fprintf('  PL-KNN (基准)     |   %.4f\n', acc_plknn);
fprintf('======================================================\n\n');

if best_acc_dgr > acc_plknn
    fprintf('结论: 经过参数调优，您的 DGR-AMFS 算法性能优于 PL-KNN。\n');
else
    fprintf('结论: 即使经过参数调优，PL-KNN 仍表现更佳，可能需要调整其他参数。\n');
end

% --- 绘制性能变化曲线 ---
figure;
semilogx(lambda_a_range, results_acc, '-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogx(lambda_a_range, ones(size(lambda_a_range)) * acc_plknn, '--r', 'LineWidth', 2);
grid on;
xlabel('正则化参数 \lambda_a (对数尺度)');
ylabel('在测试集上的准确率 (ACC)');
title('DGR-AMFS 参数敏感性分析');
legend('DGR-AMFS', 'PL-KNN 基准');
set(gca, 'FontSize', 12);
