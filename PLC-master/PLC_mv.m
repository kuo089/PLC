function final_Q = PLC_mv(X, F, plc_params)
% =========================================================================
% PLC_mv: 多视图偏标记聚类核心函数
%
% 描述:
%   本函数严格按照“逐视图独立聚类 -> 标签对齐 -> 多数投票”的思路，
%   实现一个健壮的多视图偏标记聚类算法。它被 PLC_wrapper.m 调用。
%
% 输入:
%   X          - 全量数据的特征 (cell array, V个视图)
%   F          - 初始偏标记矩阵 (n x c)
%   plc_params - 包含核心 PLC 算法所需参数的结构体
%
% 输出:
%   final_Q    - 最终的多视图共识伪标签 (n x c, one-hot)
%
% Throws:
%   如果任何子步骤失败（如单个视图PLC或验证失败），将抛出错误，
%   由上层调用者 (PLC_wrapper) 的 try-catch 块捕获。
% =========================================================================
    V = length(X);
    [n, c] = size(F);
    
    % -----------------------------------------------------------------
    % 步骤 1: 在每个视图上独立运行 PLC
    % -----------------------------------------------------------------
    fprintf('    正在为 %d 个视图独立运行PLC...\n', V);
    all_view_labels = zeros(n, V); % 用于存储每个视图的标签向量
    
    for v = 1:V
        % --- 为单次调用构建完美的输入 ---
        [~, F_gnd] = max(F, [], 2);
        Data_for_PLC = cell(1, 2);
        Data_for_PLC{1, 1} = X{v};
        Data_for_PLC{1, 2} = F_gnd;
        
        % 调用核心PLC算法，并明确告知它只处理一个视图
        Q_v_onehot = PLC(Data_for_PLC, F, 1, plc_params);
        
        % 验证结果
        if size(Q_v_onehot, 1) ~= n || any(sum(abs(Q_v_onehot), 2) < 1e-6)
            error('PLC在视图 %d 上未能为所有样本返回有效标签。', v);
        end
        
        % 存储该视图的标签结果
        [~, all_view_labels(:, v)] = max(Q_v_onehot, [], 2);
    end
    fprintf('    所有视图PLC运行完毕。\n');

    % -----------------------------------------------------------------
    % 步骤 2: 标签对齐 (Label Alignment)
    % -----------------------------------------------------------------
    fprintf('    正在对齐各视图的标签...\n');
    % 以第一个视图的标签为基准 (reference)
    ref_labels = all_view_labels(:, 1);
    
    for v = 2:V
        current_labels = all_view_labels(:, v);
        % 使用 bestMap 进行对齐
        aligned_labels = bestMap(ref_labels, current_labels);
        % 更新该视图的标签为对齐后的标签
        all_view_labels(:, v) = aligned_labels;
    end
    
    % -----------------------------------------------------------------
    % 步骤 3: 多数投票 (Majority Voting)
    % -----------------------------------------------------------------
    fprintf('    正在进行多数投票...\n');
    % mode 函数会找到每一行的众数，即得票最多的标签
    final_labels = mode(all_view_labels, 2);

    % -----------------------------------------------------------------
    % 步骤 4: 转换为 one-hot 格式输出
    % -----------------------------------------------------------------
    final_Q = full(sparse(1:n, final_labels, 1, n, c));
end
