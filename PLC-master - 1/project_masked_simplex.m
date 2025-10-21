function P = project_masked_simplex(V, Y)
% =========================================================================
% 将矩阵 V 的每一行投影到由掩码 Y 定义的单纯形上
% V: 待投影的矩阵 (n x c)
% Y: 偏标记掩码矩阵 (n x c)，值为 0 或 1
% P: 投影后的矩阵 (n x c)，满足行和为1，非负，且 P(i,j)=0 if Y(i,j)=0
% =========================================================================
[n, c] = size(V);
P = zeros(n, c);

for i = 1:n
    v_i = V(i, :);
    y_i = Y(i, :);
    
    active_indices = find(y_i == 1);
    if isempty(active_indices)
        continue; % 如果没有候选标签，则该行为0
    end
    
    v_active = v_i(active_indices);
    
    % 经典单纯形投影算法 (Duchi et al., 2008)
    u = sort(v_active, 'descend');
    cssv = cumsum(u);
    rho = find(u > (cssv - 1) ./ (1:length(u)), 1, 'last');
    theta = (cssv(rho) - 1) / rho;
    
    p_active = max(v_active - theta, 0);
    
    P(i, active_indices) = p_active;
end

end
