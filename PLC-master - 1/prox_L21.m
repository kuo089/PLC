function U_prox = prox_L21(U, lambda)
% =========================================================================
% L2,1 范数的近端算子 (行稀疏)
% U: 输入矩阵
% lambda: 正则化参数
% =========================================================================
U_prox = zeros(size(U));
for i = 1:size(U, 1)
    row = U(i, :);
    norm_row = norm(row, 2);
    if norm_row > 0
        % L2,1 proximal operator
        shrinkage = max(0, 1 - lambda / norm_row);
        U_prox(i, :) = shrinkage * row;
    end
end
end

