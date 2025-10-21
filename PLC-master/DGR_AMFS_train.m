function [F, U, alpha, obj_values] = DGR_AMFS_train(X, Y, params)
% =========================================================================
% DGR-AMFS 核心训练函数 (最终修正版)
% 包含交替优化循环
% =========================================================================

% --- 初始化 ---
[n, ~] = size(X{1});
c = params.c;
V = params.V;
obj_values = [];

% 初始化 F: 在候选标签内均匀分布
F = Y ./ sum(Y, 2);
F(isnan(F)) = 1/c; % 处理分母为0的情况
F_old = F;

% 初始化 U
U = cell(1, V);
for v = 1:V
    [~, dv] = size(X{v});
    U{v} = rand(dv, c) * 0.01;
end

% 初始化 alpha
alpha = ones(V, 1) / V;

% 初始化图, Q 等
L_bar = [];
L_cons = [];
Q = [];
s_i = ones(n, 1);
c_bar = 1;

fprintf('%-5s | %-12s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n', ...
    'Iter', 'Obj Value', 'Term1', 'Term2', 'Term3', 'Term4', 'Term5', 'Term6');
fprintf(repmat('-', 1, 90));
fprintf('\n');

% --- 交替优化主循环 ---
for iter = 1:params.max_iter
    
    % --- 1. 更新图① (特征流形图) ---
    if mod(iter, params.r_G1) == 1 || iter == 1
        fprintf('  (图①刷新)\n');
        [L_bar] = construct_graphs(X, alpha, s_i, params, 'G1');
    end

    % --- 2. B路: 调用PLC生成外生伪标签Q ---
    if mod(iter, params.m_PLC) == 1 || iter == 1
        fprintf('  (B路刷新) 调用 PLC 生成 Q...\n');
        Q = PLC_wrapper(X, F, params);
    end
    
    % --- 3. 对齐并计算一致性 ---
    [Q_aligned, ~, s_i, c_bar] = align_and_get_consistency(F, Q);
    
    % --- 4. 更新图② (共识图) ---
    [~, L_cons] = construct_graphs(X, alpha, s_i, params, 'G2');
    
    % --- 5. A路: 联合更新 F, U, alpha ---
    is_warmup = (iter <= params.warmup_iter);
    if is_warmup
        current_lambda_a = 0;
        current_lambda_B = 0;
        current_lambda_e = 0;
        fprintf('  (预热轮 %d/%d)\n', iter, params.warmup_iter);
    else
        current_lambda_a = params.lambda_a;
        current_lambda_B = params.lambda_B;
        current_lambda_e = params.lambda_e;
    end

    % [!! 最终修复 !!] 采用加权输入的方式来调用 update_U，以匹配旧版函数的接口
    % 更新 U (特征选择矩阵)
    for v = 1:V
        % 将视图权重 alpha(v) 的影响预先乘到数据和标签中
        weight = sqrt(alpha(v));
        X_v_weighted = weight * X{v};
        F_weighted = weight * F;
        
        % 现在只用4个参数调用 update_U，这应该能与您代码库中的版本兼容
        U{v} = update_U(X_v_weighted, F_weighted, U{v}, params.lambda_s);
    end

    % 更新 F (内生伪标签)
    F = update_F(F, X, U, Y, L_bar, L_cons, Q_aligned, alpha, s_i, ...
                 params.beta, current_lambda_a, current_lambda_B);

    % 更新 alpha (视图权重)
    alpha = update_alpha(X, U, F, alpha, c_bar, current_lambda_e, params.gamma);
    
    % --- 6. 计算目标函数值并检查收敛 ---
    term1 = 0;
    for v = 1:V
        term1 = term1 + alpha(v) * norm(X{v} * U{v} - F, 'fro')^2;
    end
    % 计算L2,1范数需要将所有U矩阵拼接起来
    U_concat = [];
    for v = 1:V
        U_concat = [U_concat; U{v}];
    end
    term2 = params.lambda_s * sum(sqrt(sum(U_concat.^2, 2)));

    term3 = params.beta * trace(F' * L_bar * F);
    term4 = current_lambda_a * (1/n) * sum(s_i .* sum((F - Q_aligned).^2, 2));
    term5 = current_lambda_B * trace(F' * L_cons * F);
    term6 = current_lambda_e * (c_bar^params.gamma) * sum(alpha .* log(alpha + eps));
    
    obj_val = term1 + term2 + term3 + term4 + term5 + term6;
    obj_values = [obj_values, obj_val];
    
    % 增加 NaN 安全检查
    if isnan(obj_val)
        fprintf(2, '错误: 目标函数值变为 NaN，训练不稳定，提前终止。\n');
        break;
    end
    
    fprintf('%-5d | %-12.4e | %-10.4e | %-10.4e | %-10.4e | %-10.4e | %-10.4e | %-10.4e\n', ...
        iter, obj_val, term1, term2, term3, term4, term5, term6);
    
    % 检查收敛
    if iter > 1 && abs(obj_values(end) - obj_values(end-1)) / abs(obj_values(end-1)) < 1e-4
        fprintf('目标函数已收敛，停止迭代。\n');
        break;
    end
    F_old = F;
end

end

