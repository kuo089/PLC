function W = constructW_cai(X, options)
% =========================================================================
% 使用高斯核(HeatKernel)或二进制权重构建邻接矩阵 W
% X: d x n 数据矩阵 (每列是一个样本)
% options: 参数结构体, e.g., options.k, options.t, options.WeightMode
% =========================================================================
    k = options.k;
    t = options.t;
    [~, n] = size(X);
    
    D = EuDist2(X', X'); % 计算欧氏距离平方
    
    if isfield(options, 'WeightMode') && strcmp(options.WeightMode, 'HeatKernel')
        % Heat Kernel Weighting
        W = exp(-D / (2*t^2));
    else
        % Binary Weighting (kNN graph)
        [~, idx] = sort(D, 2);
        W = zeros(n, n);
        for i = 1:n
            W(i, idx(i, 2:k+1)) = 1;
        end
    end
    
    % 确保图是无向的 (对称)
    W = (W + W') / 2;
    W(1:n+1:end) = 0; % 移除自环
end

function D = EuDist2(A, B)
% 计算 A 和 B 中行向量之间的欧氏距离平方
    D = sum(A.^2, 2) * ones(1, size(B, 1)) + ...
        ones(size(A, 1), 1) * sum(B.^2, 2)' - ...
        2 * (A * B');
    D(D < 0) = 0; % 处理数值精度问题
end
