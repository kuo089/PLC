% =========================================================================
% 对比实验: DGR-AMFS vs. PL-KNN
% 描述:
%   本脚本严格遵循半监督学习的实验框架，对 DGR-AMFS 算法和
%   经典的 PL-KNN 算法进行性能比较。
%
% 流程:
%   1. 加载 MSRCv2 数据。
%   2. 按指定比例将数据划分为训练集和测试集。
%   3. 在训练集上训练/运行两种算法。
%   4. 在测试集上评估两种算法的准确率 (ACC)。
%   5. 打印并比较最终结果。
%
% 作者: 王阔 (思路) & Gemini (实现)
% 日期: 2025-10-21
% =========================================================================

clear;
clc;
close all;

% --- 路径设置 ---
addpath(genpath('./')); % 确保所有必需的函数都在路径中

%% 1. 数据加载与准备 (建立公平的赛场)
fprintf('1. 加载 MSRCv2 数据并划分训练/测试集...\n');
load('MSRCv2_Sample.mat'); 

% --- 数据格式转换 ---
Y_full = partial_target'; % 完整偏标记矩阵 (n x c)
[~, true_labels_indices] = max(target, [], 1);
true_labels_full = true_labels_indices'; % 完整真实标签 (n x 1)
[n, c] = size(Y_full);

% 模拟多视图数据
X_full = cell(1, 2);
X_full{1} = zscore(data); % 视图1
X_full{2} = zscore(data + 0.1 * randn(size(data))); % 视图2
V = length(X_full);

% --- 划分训练集与测试集 ---
train_ratio = 0.8; % 使用 80% 的数据作为训练集
indices = randperm(n);
train_idx = indices(1:floor(n * train_ratio));
test_idx = indices(floor(n * train_ratio)+1:end);

% 创建训练集
X_train = cell(1, V);
for v = 1:V
    X_train{v} = X_full{v}(train_idx, :);
end
Y_train = Y_full(train_idx, :);
true_labels_train = true_labels_full(train_idx); % (仅用于调试，算法不应使用)

% 创建测试集
X_test = cell(1, V);
for v = 1:V
    X_test{v} = X_full{v}(test_idx, :);
end
true_labels_test = true_labels_full(test_idx); % 这是我们最终的评判标准

fprintf('数据准备完毕: %d 个训练样本, %d 个测试样本。\n\n', length(train_idx), length(test_idx));


%% 2. 运行您的 DGR-AMFS 算法 (直推式学习)
fprintf('2. 开始训练和评估 DGR-AMFS 模型...\n');

% --- 设置超参数 (与 main_DGR_AMFS.m 中保持一致) ---
params.V = V;
params.c = c;
params.max_iter = 30;
params.warmup_iter = 5;
params.lambda_s = 1e-3;
params.beta = 1e-2;
params.lambda_a = 1e-1;
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

% --- 准备DGR-AMFS的输入 ---
% DGR-AMFS是直推式的，它需要看到所有样本的特征，但只能看到训练样本的偏标记
Y_for_dgr = zeros(n, c);
Y_for_dgr(train_idx, :) = Y_train; % 只提供训练集的偏标记

% --- 运行算法 ---
[F, U, alpha, ~] = DGR_AMFS_train(X_full, Y_for_dgr, params);

% --- 在测试集上评估 ---
F_test = F(test_idx, :); % 提取测试集样本的预测结果
[~, pred_labels_dgr_raw] = max(F_test, [], 2);

% 使用 bestMap 进行标签对齐 (聚类评估的标准做法)
pred_labels_dgr = bestMap(true_labels_test, pred_labels_dgr_raw);
acc_dgr = sum(pred_labels_dgr == true_labels_test) / length(test_idx);

fprintf('DGR-AMFS 模型评估完成。\n\n');


%% 3. 运行 PL-KNN 基准算法 (归纳式学习)
fprintf('3. 开始训练和评估 PL-KNN 模型...\n');

% --- 设置超参数 ---
k_knn = 10; % K-近邻的 K 值

% --- 运行算法 (训练和预测) ---
pred_labels_plknn = PLKNN_predict(X_train, Y_train, X_test, k_knn);

% --- 在测试集上评估 ---
acc_plknn = sum(pred_labels_plknn == true_labels_test) / length(test_idx);

fprintf('PL-KNN 模型评估完成。\n\n');


%% 4. 显示最终对比结果
fprintf('==================== 最终对比结果 ====================\n');
fprintf('  数据集: MSRCv2\n');
fprintf('  训练集比例: %.2f\n', train_ratio);
fprintf('------------------------------------------------------\n');
fprintf('  算法名称          | 在测试集上的ACC\n');
fprintf('------------------------------------------------------\n');
fprintf('  DGR-AMFS (您的)   |   %.4f\n', acc_dgr);
fprintf('  PL-KNN (基准)     |   %.4f\n', acc_plknn);
fprintf('======================================================\n\n');

if acc_dgr > acc_plknn
    fprintf('结论: 在本次运行中，您的 DGR-AMFS 算法性能优于 PL-KNN。\n');
else
    fprintf('结论: 在本次运行中，PL-KNN 算法性能优于您的 DGR-AMFS。\n');
end
