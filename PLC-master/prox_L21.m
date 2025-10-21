function W_new = prox_L21(W, lambda)
% =========================================================================
% L2,1 范数的近端算子 (Proximal Operator for L2,1 norm)
%
% 解决以下优化问题:
%   min_X { 1/2 * ||X - W||_F^2 + lambda * ||X||_{2,1} }
%
% 输入:
%   W: 待处理的矩阵 (d x c)
%   lambda: 正则化参数
%
% 输出:
%   W_new: 经过 L2,1 范数收缩后的矩阵
%
% *** 错误修复: 函数定义现在接收 2 个输入参数以匹配调用 ***
% =========================================================================

W_new = zeros(size(W));
for i = 1:size(W, 1) % 对 W 的每一行进行操作
    row_norm = norm(W(i, :), 2);
    if row_norm > lambda
        shrinkage_factor = (row_norm - lambda) / row_norm;
        W_new(i, :) = shrinkage_factor * W(i, :);
    else
        % 如果行范数小于 lambda，则该行变为零向量
        W_new(i, :) = 0;
    end
end

end

