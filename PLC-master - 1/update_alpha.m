function alpha_new = update_alpha(X, U, F, alpha, c_bar, lambda_e, gamma)
% =========================================================================
% 更新视图权重 alpha (α-step)
% 使用带熵正则的乘法更新/softmin
% =========================================================================
V = length(X);
e = zeros(V, 1);

% 计算每个视图的重构误差
for v = 1:V
    e(v) = norm(X{v} * U{v} - F, 'fro')^2;
end

% 带熵正则的 softmin/乘法更新规则
% 温度由全局一致性 c_bar 调节
temp = lambda_e * (c_bar^gamma) + 1e-10; % 防止 temp 为 0

% 更新 alpha
numerator = exp(-e / temp);
alpha_new = numerator / sum(numerator);

% 防止权重塌缩到 0
epsilon = 1e-4;
if any(alpha_new < epsilon)
    alpha_new = max(alpha_new, epsilon);
    alpha_new = alpha_new / sum(alpha_new);
end

end
