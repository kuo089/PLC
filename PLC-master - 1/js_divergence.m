function D_js = js_divergence(P, Q)
% =========================================================================
% 计算两个概率分布矩阵 P 和 Q 每一行之间的 Jensen-Shannon 散度
% P, Q: n x c 的矩阵，每行是一个概率分布
% =========================================================================

% 加上一个很小的平滑项，防止 log(0)
epsilon = 1e-10;
P = P + epsilon;
Q = Q + epsilon;

M = 0.5 * (P + Q);

% Kullback-Leibler 散度
D_kl_pm = sum(P .* log(P ./ M), 2);
D_kl_qm = sum(Q .* log(Q ./ M), 2);

D_js = 0.5 * (D_kl_pm + D_kl_qm);

end
