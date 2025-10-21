function [Q_aligned, M, s_i, c_bar] = align_and_get_consistency(F, Q)
% =========================================================================
% 对齐B路伪标签Q到A路伪标签F，并计算一致性
% =========================================================================
[n, c] = size(F);

% --- 1. 构建代价矩阵 (或称混淆/关联矩阵) ---
% C(i, j) 表示 F 中第 i 类 与 Q 中第 j 类共同出现的样本数
[~, F_hard] = max(F, [], 2);
[~, Q_hard] = max(Q, [], 2);
cost_matrix = zeros(c, c);
for i = 1:c
    for j = 1:c
        cost_matrix(i, j) = sum(F_hard == i & Q_hard == j);
    end
end

% --- 2. 使用匈牙利算法求解最优映射 ---
% hungarian 函数需要代价越大越好，所以我们传入负的代价矩阵
% c_map 是一个 c x 1 的向量，c_map(j) = i 表示 Q 的第 j 类映射到 F 的第 i 类
[c_map, ~] = hungarian(-cost_matrix);

% 构建置换矩阵 M
M = full(sparse(c_map, 1:c, 1, c, c));

% --- 3. 对齐 Q ---
Q_aligned = Q * M'; % M' 将 Q 的列按 c_map 重新排列

% --- 4. 计算样本级一致性 s_i 和全局一致性 c_bar ---
% 使用 Jensen-Shannon 散度
js_vals = js_divergence(F, Q_aligned); % n x 1 向量

% 全局一致性 c_bar (1 - 平均散度)
c_bar = 1 - mean(js_vals);
c_bar = max(0, c_bar); % 确保非负

% 样本门控 s_i (1 - 散度)
s_i = 1 - js_vals;
s_i = max(0, s_i); % 确保非负

% (可选) 可以使用 sigmoid 函数平滑门控，避免硬阈值
% beta_s = 10; % sigmoid 的陡峭度
% s_i = 1 ./ (1 + exp(-beta_s * (s_i - 0.5)));

end
