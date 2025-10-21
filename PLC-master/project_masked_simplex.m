function P = project_masked_simplex(V, Y)
% =========================================================================
% 将 V 的每一行投影到由掩码矩阵 Y 定义的概率单纯形上。
% 即，对于每一行 i，我们求解:
%   min_{p_i} ||p_i - v_i||^2
%   s.t.  sum(p_i) = 1, p_i >= 0, and p_i(j) = 0 if Y(i,j) = 0
%
% 输入:
%   V - 待投影的矩阵 (n x c)
%   Y - 偏标记掩码矩阵 (n x c)，其中 Y(i,j)=1 表示 j 是 i 的候选标签
%
% 输出:
%   P - 投影后的矩阵 (n x c)
% =========================================================================
[n, c] = size(V);
P = zeros(n, c);

for i = 1:n
    active_indices = find(Y(i, :) > 0);
    
    % [!! 修复 !!] 增加安全检查，处理一个样本没有任何候选标签的极端情况
    if isempty(active_indices)
        % 如果没有候选标签（理论上不应发生），则赋予所有类别均匀概率
        % 这可以防止程序因 active_indices 为空而崩溃
        P(i, :) = 1/c;
        continue; % 处理下一个样本
    end
    
    v_active = V(i, active_indices);
    k = length(active_indices);
    
    % 使用高效的单纯形投影算法 (Duchi et al., 2008)
    u = sort(v_active, 'descend');
    cssv = cumsum(u);
    rho = find(u > (cssv - 1) ./ (1:k), 1, 'last');
    
    if isempty(rho)
        % 这种情况很少见，但可能发生
        % 意味着 v_active 的所有元素都非常小或为负
        % 我们选择将所有权重放在最大的那个元素上
        [~, max_idx] = max(v_active);
        p_active = zeros(1, k);
        p_active(max_idx) = 1;
    else
        theta = (cssv(rho) - 1) / rho;
        p_active = max(v_active - theta, 0);
    end
    
    P(i, active_indices) = p_active;
end

end

