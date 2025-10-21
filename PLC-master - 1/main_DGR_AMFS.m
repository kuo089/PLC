% =========================================================================
% 面向偏标记学习的双图正则化自适应多视图特征选择 (DGR-AMFS)
% 主运行文件 (main_DGR_AMFS.m)
% 作者：王阔 (思路) & Gemini (实现)
% 日期：2025-10-21
% =========================================================================
clear;
clc;
close all;

% --- 添加路径 ---
addpath(genpath('core'));
addpath(genpath('utils'));
addpath(genpath('external')); % 假设原始PLC代码放在external文件夹

fprintf('开始执行 DGR-AMFS 算法...\n');

% --- 1. 数据加载与准备 ---
% 在这里加载您的多视图偏标记数据集
% MVP 阶段，我们生成一个合成数据集用于演示
fprintf('1. 生成合成多视图偏标记数据...\n');
n = 200;    % 样本数
d1 = 50;    % 视图1特征维度
d2 = 60;    % 视图2特征维度
c = 5;      % 类别数
V = 2;      % 视图数

% 生成清晰的聚类数据
data1 = rand(n, d1);
data2 = rand(n, d2);
true_labels = repelem(1:c, n/c)'; % 真实标签
X{1} = data1;
X{2} = data2;
for i = 1:c
    idx = (i-1)*(n/c)+1 : i*(n/c);
    X{1}(idx, 1:10) = X{1}(idx, 1:10) + 1.5; % 让前10个特征具有区分性
    X{2}(idx, 1:10) = X{2}(idx, 1:10) + 1.8;
end

% 生成偏标记矩阵 Y
% 协议：真实标签 + 从其余 c-1 类中随机抽取 rho*(c-1) 个干扰标签
rho = 0.2; % 偏标记率
Y = zeros(n, c);
for i = 1:n
    true_label = true_labels(i);
    Y(i, true_label) = 1;
    other_labels = find((1:c) ~= true_label);
    num_noise = floor(rho * (c-1));
    if num_noise > 0
        noise_indices = randperm(length(other_labels), num_noise);
        Y(i, other_labels(noise_indices)) = 1;
    end
end
fprintf('数据准备完毕. %d 个样本, %d 个视图, %d 个类别.\n\n', n, V, c);

% --- 2. 设置超参数 ---
fprintf('2. 设置超参数...\n');
params.c = c;
params.V = V;

% 正则化参数 (对应目标函数)
params.lambda_s = 1e-3;   % ② L2,1 稀疏项
params.beta = 1e-2;       % ③ 图① (特征流形)
params.lambda_a = 1e-1;   % ④ A<->B 对齐一致性
params.lambda_B = 1e-2;   % ⑤ 图② (共识图)
params.lambda_e = 1e-2;   % ⑥ 一致性驱动的视图熵
params.gamma = 2;         % 熵正则的指数

% 训练流程参数
params.max_iter = 30;     % 主循环最大迭代次数
params.warmup_iter = 5;   % 预热轮数
params.r_G1 = 3;          % 图① 重建频率
params.m_PLC = 5;         % B路(PLC) 刷新频率

% 图构建参数
params.k_G1 = 10;         % 图① 的 kNN 参数
params.k_G2 = 10;         % 图② 的 kNN 参数
params.sigma = 1;         % 高斯核的 sigma

% PLC 算法的参数 (从您的 PLC_demo.m 中提取)
plc_para.alpha=0.01;
plc_para.beta=0.01;
plc_para.gamma=10;
plc_para.lambda=1;
plc_para.mu=1;
plc_para.maxiter=10; % AdversarialPCP 的迭代
plc_para.maxit=5;    % PLC 主循环的迭代
plc_para.k=10;
params.plc_para = plc_para;
fprintf('超参数设置完毕.\n\n');

% --- 3. 运行算法 ---
fprintf('3. 开始训练 DGR-AMFS 模型...\n');
[F, U, alpha, obj_values] = DGR_AMFS_train(X, Y, params);
fprintf('模型训练完成.\n\n');

% --- 4. 结果评估 ---
fprintf('4. 评估结果...\n');
[~, predict_labels] = max(F, [], 2);

% 使用 bestMap (来自您提供的文件) 进行标签对齐
addpath(genpath('external')); % 确保 bestMap 在路径中
aligned_labels = bestMap(true_labels, predict_labels);

% 计算 ACC
acc = sum(aligned_labels == true_labels) / n;
fprintf('最终聚类 ACC: %.4f\n', acc);

% 绘制目标函数值变化曲线
figure;
plot(1:length(obj_values), obj_values, '-o');
xlabel('迭代次数');
ylabel('目标函数值');
title('DGR-AMFS 目标函数值收敛曲线');
grid on;

% 显示视图权重
figure;
bar(alpha);
xlabel('视图索引');
ylabel('权重');
title('最终视图权重 α');
xticklabels(arrayfun(@(i) sprintf('View %d', i), 1:V, 'UniformOutput', false));

fprintf('程序运行结束。\n');
