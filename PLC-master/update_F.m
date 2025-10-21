function F_new = update_F(F, X, U, Y, L_bar, L_cons, Q_aligned, alpha, s_i, beta, lambda_a, lambda_B)
% =========================================================================
% 更新内生伪标签矩阵 F (F-step)
% 采用梯度下降 + 掩码单纯形投影
% =========================================================================
n = size(F, 1);
V = length(X);

% --- 计算目标函数关于 F 的梯度 ---
% 对应目标函数 ①, ③, ④, ⑤

% Term ①: 多视图拟合项的梯度
grad_term1 = zeros(size(F));
XU_sum = zeros(size(F));
for v = 1:V
    XU_sum = XU_sum + alpha(v) * (X{v} * U{v});
end
grad_term1 = 2 * (F - XU_sum);

% Term ③: 图①正则项的梯度
grad_term3 = 2 * beta * L_bar * F;

% [!! 修复 !!] Term ④: A<->B 对齐一致性损失的梯度
% 使用在数值上更稳定的平方 L2 范数 ||F - Q_aligned||^2 来替代 JS 散度。
% 这可以有效避免由 log 函数导致的 NaN 问题。
grad_term4 = 2 * lambda_a * (1/n) * (s_i .* (F - Q_aligned)); 

% Term ⑤: 图②正则项的梯度
grad_term5 = 2 * lambda_B * L_cons * F;

% 总梯度
grad_F = grad_term1 + grad_term3 + grad_term4 + grad_term5;

% --- 梯度下降 + 投影 ---
% 使用一个相对保守的固定步长
step_size = 0.05; 
F_updated = F - step_size * grad_F;

% 投影到满足偏标记约束的单纯形上
F_new = project_masked_simplex(F_updated, Y);

end
