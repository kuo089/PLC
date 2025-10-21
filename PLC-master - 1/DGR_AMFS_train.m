function [F, U, alpha, obj_values] = DGR_AMFS_train(X, Y, params)
% =========================================================================
% DGR-AMFS 核心训练函数
% 包含交替优化循环
% =========================================================================

% --- 初始化 ---
[n, ~] = size(X{1});
c = params.c;
V = params.V;
obj_values = [];

% 初始化 F (内生伪标签): 在候选标签内均匀分布
F = Y ./ sum(Y, 2);
F_old = F;

% 初始化 U (特征选择矩阵)
U = cell(1, V);
for v = 1:V
    [~, dv] = size(X{v});
    U{v} = rand(dv, c) * 0.01;
end

% 初始化 alpha (视图权重)
alpha = ones(V, 1) / V;

% 初始化图, Q, M, s_i, c_bar 为空，在第一轮迭代中构建
L_bar = [];
L_cons = [];
Q = [];
M = [];
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
        [L_bar] = construct_graphs(X, alpha, s_i, params, 'G1');
    end

    % --- 2. B路: 调用PLC生成外生伪标签Q (间隔刷新) ---
    if mod(iter, params.m_PLC) == 1 || iter == 1
        fprintf('  (B路刷新) 调用 PLC 生成 Q...\n');
        Q = PLC_wrapper(X, F, params.c, params.plc_para);
    end
    
    % --- 3. 对齐并计算一致性 ---
    [Q_aligned, M, s_i, c_bar] = align_and_get_consistency(F, Q);
    
    % --- 4. 更新图② (共识图) ---
    % 共识图每轮都根据最新的 s_i 更新
    [~, L_cons] = construct_graphs(X, alpha, s_i, params, 'G2');
    
    % --- 5. A路: 联合更新 F, U, alpha ---
    % 在预热阶段，关闭一些正则项
    current_lambda_a = params.lambda_a;
    current_lambda_B = params.lambda_B;
    current_lambda_e = params.lambda_e;
    if iter <= params.warmup_iter
        current_lambda_a = 0;
        current_lambda_B = 0;
        current_lambda_e = 0;
        fprintf('  (预热轮 %d/%d)\n', iter, params.warmup_iter);
    end

    % 更新 U (特征选择矩阵)
    for v = 1:V
        U{v} = update_U(X{v}, F, U{v}, params.lambda_s, alpha(v));
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
    term2 = 0;
    for v = 1:V
        term2 = term2 + params.lambda_s * L21_norm(U{v});
    end
    term3 = params.beta * trace(F' * L_bar * F);
    
    js_vals = js_divergence(F, Q_aligned);
    term4 = current_lambda_a * (1/n) * sum(s_i .* js_vals);

    term5 = current_lambda_B * trace(F' * L_cons * F);
    term6 = current_lambda_e * (c_bar^params.gamma) * sum(alpha .* log(alpha + 1e-10));
    
    obj_val = term1 + term2 + term3 + term4 + term5 + term6;
    obj_values = [obj_values, obj_val];
    
    fprintf('%-5d | %-12.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %-10.4f\n', ...
        iter, obj_val, term1, term2, term3, term4, term5, term6);
    
    % 检查收敛
    if iter > 1 && norm(F - F_old, 'fro') / norm(F_old, 'fro') < 1e-4
        fprintf('F 矩阵已收敛，停止迭代。\n');
        break;
    end
    F_old = F;
end

end


