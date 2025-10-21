function [L_bar, L_cons] = construct_graphs(X, alpha, s_i, params, mode)
% =========================================================================
% 构建图① (L_bar) 和/或 图② (L_cons)
% =========================================================================
n = size(X{1}, 1);
V = params.V;
k_G1 = params.k_G1;
k_G2 = params.k_G2;
sigma = params.sigma;
L_bar = [];
L_cons = [];

% --- 构建图①：特征流形图 ---
if strcmp(mode, 'G1') || strcmp(mode, 'all')
    L_sum = zeros(n, n);
    for v = 1:V
        options.k = k_G1;
        options.WeightMode = 'HeatKernel';
        options.t = sigma;
        % 调用独立的 constructW_cai 函数
        A_v = constructW_cai(X{v}', options); 
        
        D_v_sum = sum(A_v, 2);
        D_v_inv_sqrt = diag(1 ./ sqrt(D_v_sum + 1e-10));
        L_v = eye(n) - D_v_inv_sqrt * A_v * D_v_inv_sqrt;
        L_sum = L_sum + alpha(v) * L_v;
    end
    L_bar = L_sum;
end

% --- 构建图②：共识图 ---
if strcmp(mode, 'G2') || strcmp(mode, 'all')
    % 找到高一致性、高置信度的样本子集
    confidence_threshold = 0.7; % 这是一个需要调整的超参数
    high_conf_idx = find(s_i > confidence_threshold);
    
    if length(high_conf_idx) < k_G2 + 1
        % 如果高置信度样本太少，则不构建共识图
        L_cons = zeros(n, n);
    else
        X_concat_conf = [];
        for v = 1:V
             X_concat_conf = [X_concat_conf, X{v}(high_conf_idx, :)];
        end
        
        options.k = k_G2;
        options.WeightMode = 'HeatKernel';
        options.t = sigma;
        % 调用独立的 constructW_cai 函数
        W_cons_sub = constructW_cai(X_concat_conf', options);
        
        % 根据 s_i 调整权重
        s_i_sub = s_i(high_conf_idx);
        W_cons_sub = (s_i_sub * s_i_sub') .* W_cons_sub;
        
        D_cons_sub_sum = sum(W_cons_sub, 2);
        D_cons_sub_inv_sqrt = diag(1 ./ sqrt(D_cons_sub_sum + 1e-10));
        L_cons_sub = eye(length(high_conf_idx)) - D_cons_sub_inv_sqrt * W_cons_sub * D_cons_sub_inv_sqrt;
        
        % 将子图的拉普拉斯矩阵扩展回完整大小
        L_cons = zeros(n, n);
        L_cons(high_conf_idx, high_conf_idx) = L_cons_sub;
    end
end

end

