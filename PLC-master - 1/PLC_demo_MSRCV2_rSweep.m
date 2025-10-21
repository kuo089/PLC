% 加载数据
load('MSRCv2_Sample.mat'); % 假设这个.mat文件包含 'data' 和 'target' 变量
                          % 'data' 是特征矩阵 (样本数 x 特征数)
                          % 'target' 是真实标签矩阵 (类别数 x 样本数)，通常用于生成 partial_target

% 定义参数 (保持不变)
para.alpha=0.01;
para.beta=0.01;
para.gamma=10;
para.lambda=1;
para.mu=1;
para.maxiter=10; % AdversarialPCP的最大迭代次数
para.maxit=5;    % PLC主循环的最大迭代次数
para.k=10;

% 定义不同的rho值 (部分标记样本的比例)
rho_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40];
num_rho = length(rho_values);
num_folds = 5; % 将折数降低到5

% 初始化存储结果的变量
results_acc = zeros(num_rho, num_folds);
results_nmi = zeros(num_rho, num_folds);
mean_acc_per_rho = zeros(num_rho, 1);
std_acc_per_rho = zeros(num_rho, 1);
mean_nmi_per_rho = zeros(num_rho, 1);
std_nmi_per_rho = zeros(num_rho, 1);

num_samples = size(data, 1);
num_classes = size(target, 1);

fprintf('开始实验...\n');

% 循环不同的rho值
for r_idx = 1:num_rho
    rho = rho_values(r_idx);
    num_train = round(num_samples * rho);
    num_test = num_samples - num_train;

    fprintf('当前 rho = %.2f (训练样本数: %d, 测试样本数: %d)\n', rho, num_train, num_test);

    ACC_fold = [];
    NMI_fold = [];

    % 进行5折交叉验证
    for fold = 1:num_folds
        fprintf('  Fold %d/%d:\n', fold, num_folds);

        % 随机划分训练集和测试集索引
        indices = randperm(num_samples);
        train_idx_fold = indices(1:num_train);
        test_idx_fold = indices(num_train+1:end);

        train_data = data(train_idx_fold, :);
        train_data = zscore(train_data); % Z-score 标准化训练数据
        test_data = data(test_idx_fold, :);
        test_data = zscore(test_data);   % Z-score 标准化测试数据

        % 生成部分标签 train_p_target
        % 这里需要根据原始数据和'target'生成部分标签
        % 论文中提到使用 [Cour et al., 2011] 的方法生成，参数为 r (false-positive 标签数)
        % 这里我们假设 r=1 (可以根据需要修改)
        r_false_positive = 1; % 每个样本添加 r 个错误候选标签
        train_target_full = target(:, train_idx_fold); % 获取训练样本的真实标签 (one-hot)
        train_p_target = zeros(size(train_target_full));
        [~, true_labels_idx] = max(train_target_full, [], 1); % 获取真实标签的索引

        for i = 1:num_train
            true_label = true_labels_idx(i);
            train_p_target(true_label, i) = 1; % 添加真实标签

            % 添加 r 个错误标签
            possible_false_labels = find((1:num_classes) ~= true_label);
            false_label_indices = randperm(length(possible_false_labels), min(r_false_positive, length(possible_false_labels)));
            false_labels = possible_false_labels(false_label_indices);
            train_p_target(false_labels, i) = 1;
        end

        test_target = target(:, test_idx_fold); % 测试集的真实标签

        % 运行PLC算法
        fprintf('    运行 PLC 算法...\n');
        groups = PLC(train_data, train_p_target, test_data, para);
        fprintf('    PLC 运行完毕.\n');

        % 计算指标
        [acc, nmi] = CalMetrics(test_target, groups);
        fprintf('    Fold %d ACC: %f, NMI: %f\n', fold, acc, nmi);

        ACC_fold = [ACC_fold, acc];
        NMI_fold = [NMI_fold, nmi];
    end

    % 存储当前rho值的结果
    results_acc(r_idx, :) = ACC_fold;
    results_nmi(r_idx, :) = NMI_fold;

    % 计算当前rho值的平均值和标准差
    mean_acc_per_rho(r_idx) = mean(ACC_fold);
    std_acc_per_rho(r_idx) = std(ACC_fold);
    mean_nmi_per_rho(r_idx) = mean(NMI_fold);
    std_nmi_per_rho(r_idx) = std(NMI_fold);

    fprintf('rho = %.2f 的平均 ACC: %f (标准差: %f)\n', rho, mean_acc_per_rho(r_idx), std_acc_per_rho(r_idx));
    fprintf('rho = %.2f 的平均 NMI: %f (标准差: %f)\n\n', rho, mean_nmi_per_rho(r_idx), std_nmi_per_rho(r_idx));
end

fprintf('实验完成。\n\n');

% 显示最终结果
fprintf('不同 rho 值的平均 ACC 和标准差:\n');
for r_idx = 1:num_rho
    fprintf('rho = %.2f: ACC = %f +/- %f\n', rho_values(r_idx), mean_acc_per_rho(r_idx), std_acc_per_rho(r_idx));
end
fprintf('\n');

fprintf('不同 rho 值的平均 NMI 和标准差:\n');
for r_idx = 1:num_rho
    fprintf('rho = %.2f: NMI = %f +/- %f\n', rho_values(r_idx), mean_nmi_per_rho(r_idx), std_nmi_per_rho(r_idx));
end
fprintf('\n');

% 分析ACC随rho变化的趋势
fprintf('分析 ACC 随 rho 变化的趋势:\n');
acc_diff = diff(mean_acc_per_rho); % 计算相邻rho值的ACC差值
if all(acc_diff >= 0)
    fprintf('ACC 随着 rho 的增大而增大或保持不变。\n');
elseif all(acc_diff <= 0)
    fprintf('ACC 随着 rho 的增大而减小或保持不变。\n');
else
    fprintf('ACC 随 rho 变化的趋势不单调。\n');
    % 可以进一步分析具体变化情况
    for i = 1:length(acc_diff)
        if acc_diff(i) > 0
            fprintf('  从 rho=%.2f 到 rho=%.2f, ACC 增大。\n', rho_values(i), rho_values(i+1));
        elseif acc_diff(i) < 0
            fprintf('  从 rho=%.2f 到 rho=%.2f, ACC 减小。\n', rho_values(i), rho_values(i+1));
        else
            fprintf('  从 rho=%.2f 到 rho=%.2f, ACC 保持不变。\n', rho_values(i), rho_values(i+1));
        end
    end
end

% 可以选择绘制图形
% figure;
% errorbar(rho_values, mean_acc_per_rho, std_acc_per_rho, '-o');
% xlabel('部分标记样本比例 (ρ)');
% ylabel('平均 ACC');
% title('ACC 随 ρ 变化的趋势 (5折交叉验证)');
% grid on;