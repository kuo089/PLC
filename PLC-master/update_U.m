function U_new = update_U(X, F, U_init, lambda_s)
% =========================================================================
% 稳健地求解 U 的子问题: min_U ||XU - F||^2_F + lambda_s * ||U||_2,1
% 使用迭代收缩阈值算法 (Iterative Shrinkage-Thresholding Algorithm, ISTA)。
% 这个实现是自包含的，不依赖于外部的 prox_L21.m 文件。
%
% 输入:
%   X        - (加权后的) 视图数据矩阵
%   F        - (加权后的) 伪标签矩阵
%   U_init   - U 的初始值
%   lambda_s - L2,1 范数的正则化参数
%
% 输出:
%   U_new    - 优化后的 U
% =========================================================================

% --- 求解器的超参数 ---
max_iter = 20; % 此子问题的最大迭代次数
tol = 1e-4;    % 用于判断收敛的容差

% --- 计算步长 ---
% 一个安全的步长是 1/L，其中 L 是梯度算子的 Lipschitz 常数。
% L = 2 * ||X'X||_2 (即 X'X 的最大特征值)
try
    % eigs 更快，只计算最大特征值
    L = 2 * eigs(X'*X, 1);
    if L == 0 % 如果 L 为 0，设置一个小的默认值
        L = 1e-3;
    end
    step_size = 1 / L;
catch
    % 如果 X'X 太大导致计算困难，使用一个保守的小步长
    step_size = 1e-4; 
end

U_k = U_init; % 从上一次迭代的 U 值开始

for iter = 1:max_iter
    % --- 1. 梯度下降步骤 ---
    % 计算最小二乘项的梯度
    grad = 2 * X' * (X * U_k - F);
    % 沿梯度负方向更新
    U_grad = U_k - step_size * grad;
    
    % --- 2. 近端映射步骤 (处理 L2,1 范数) ---
    % 这部分逻辑直接在这里实现，替代了对 prox_L21.m 的调用
    U_new = zeros(size(U_grad));
    [d, ~] = size(U_grad);
    
    prox_lambda = step_size * lambda_s; % 应用于近端算子的 lambda
    
    for i = 1:d
        row = U_grad(i, :);
        row_norm = norm(row, 2);
        
        if row_norm > prox_lambda
            % 如果行的 L2 范数大于阈值，则按比例收缩它
            shrinkage_factor = (row_norm - prox_lambda) / row_norm;
            U_new(i, :) = shrinkage_factor * row;
        else
            % 否则，根据 L2,1 范数的性质，将整行收缩为零
            U_new(i, :) = 0;
        end
    end
    
    % --- 3. 检查收敛 ---
    if norm(U_new - U_k, 'fro') / (norm(U_k, 'fro') + eps) < tol
        break;
    end
    
    U_k = U_new;
end

end
