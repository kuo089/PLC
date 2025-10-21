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

% Term ④: A<->B 对齐一致性损失的梯度
% JS(p||q) 的梯度 w.r.t p 是复杂的，这里用一个简化形式
% 我们惩罚 ||F - Q_aligned||^2，并用 s_i 加权
% grad_term4 = 2 * lambda_a * (1/n) * (s_i .* (F - Q_aligned)); 
% 注：JS散度的梯度更精确，但实现复杂。这个平方损失是常用的替代方案。
% 为忠于原文，我们应使用 JS 散度梯度
M_fq = 0.5 * (F + Q_aligned);
grad_term4 = lambda_a * (1/n) * s_i .* (log(F + 1e-10) - log(M_fq + 1e-10));


% Term ⑤: 图②正则项的梯度
grad_term5 = 2 * lambda_B * L_cons * F;

% 总梯度
grad_F = grad_term1 + grad_term3 + grad_term4 + grad_term5;

% --- 梯度下降 + 投影 ---
step_size = 0.1; % 步长可以设为固定值或使用线搜索
F_updated = F - step_size * grad_F;

% 投影到满足偏标记约束的单纯形上
F_new = project_masked_simplex(F_updated, Y);

end
