function alpha_new = update_alpha(X, U, F, alpha_init, c_bar, lambda_e, gamma)
% =========================================================================
% 更新视图权重 alpha (alpha-step)
% 采用数值稳定的方法求解闭式解
% =========================================================================
V = length(X);
costs = zeros(V, 1);

% --- 1. 计算每个视图的代价 ---
for v = 1:V
    % 代价由两部分组成：
    % a) 来自 Term1 的数据拟合误差
    fit_error = norm(X{v} * U{v} - F, 'fro')^2;
    
    % b) 来自 Term6 的熵正则项
    % 注意：这里的 c_bar^gamma 是一个常数乘子，可以移到指数之外
    % 但为忠实于模型，我们将其作为代价的一部分
    entropy_term = lambda_e * (c_bar^gamma); 
    
    costs(v) = fit_error + entropy_term;
end

% --- 2. 使用 log-sum-exp 技巧稳定地计算 alpha ---
% 这是避免 exp() 溢出导致 NaN 的关键
% alpha_v ∝ exp(-costs(v))

% 从所有代价中减去最小值，防止 exp() 的参数过大导致 Inf
min_cost = min(costs);
temp = exp(-(costs - min_cost)); 

% 归一化得到新的 alpha
alpha_new = temp / sum(temp);

% --- 3. 安全检查，防止权重完全为 0 ---
% 确保权重不会小于 MATLAB 的最小浮点数
alpha_new(alpha_new < eps) = eps;
% 再次归一化以确保和为 1
alpha_new = alpha_new / sum(alpha_new);

end
