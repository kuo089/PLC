function convert_lost_to_sample(rho)
% 将 lost.mat 转换成 lost_Sample.mat（MSRCv2_Sample 风格）
% rho: 候选标签保留比例，例如 0.1 (保留 10%)

if nargin<1, rho=0.1; end    % 默认 ρ=0.1
rng(1);                      % 固定随机种子保证可复现

% ====== 读取原始数据 ======
S = load('lost.mat');
assert(isfield(S,'data'), 'lost.mat 必须包含变量 data');

X = S.data;
if isfield(S,'target')
    T = S.target;   % c×n
    if size(X,2)==size(T,2)   % d×n -> 转置为 n×d
        X = X.';
    end
elseif isfield(S,'labels')
    y = S.labels(:);
    if size(X,2)==numel(y)    % d×n -> 转置为 n×d
        X = X.';
    end
    n = size(X,1);
    classes = unique(y);
    C = numel(classes);
    T = zeros(C,n);
    for i=1:C
        T(i, y==classes(i))=1;
    end
else
    error('lost.mat 里必须有 target(c×n) 或 labels(n×1)');
end

[n,d] = size(X);
C = size(T,1);

% ====== 生成部分标签 ======
partial_target = zeros(C,n);
m = max(1, ceil(1/rho));   % 每个样本候选标签数
for i=1:n
    ci = find(T(:,i));
    partial_target(ci,i)=1;   % 保留真标签
    if m>1
        pool = setdiff(1:C,ci);
        k = min(m-1,numel(pool));
        if k>0
            idx = randperm(numel(pool),k);
            partial_target(pool(idx),i)=1;
        end
    end
end

% ====== 生成 10 折划分 ======
y = vec_from_onehot(T);       % n×1
K = 10;
cv = cvpartition(y,'KFold',K);
train_idx = cell(1,K);
test_idx  = cell(1,K);
for k=1:K
    train_idx{k} = find(training(cv,k));
    test_idx{k}  = find(test(cv,k));
end

% ====== 保存成 lost_Sample.mat ======
data = X; target = T;
save('lost_Sample.mat','data','target','partial_target','train_idx','test_idx');

fprintf('已生成 lost_Sample.mat，可直接用 PLC_demo.m 跑\n');
end

% ===== 辅助函数 =====
function y = vec_from_onehot(T)
[~,y] = max(T,[],1);
y = y(:);
end
