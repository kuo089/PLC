% demo_lost.m —— 简洁版
clear; clc; rng(1);

load('lost.mat');                      % 需要包含 data，以及 target(c×n) 或 labels(n×1)

% 若 data 为 d×n 且 target 为 c×n，则转成 n×d（样本在行）
if exist('target','var') && size(data,2)==size(target,2), data = data.'; end
n = size(data,1);

% 统一得到 one-hot 真值 T (c×n)
if exist('target','var')
    T = target;                                % c×n
    assert(size(T,2)==n, 'target 与 data 样本数不一致');
else
    assert(exist('labels','var')==1, '需要 target 或 labels');
    y = labels(:);
    if size(data,2)==numel(y), data = data.'; n = size(data,1); end
    assert(numel(y)==n, 'labels 与 data 样本数不一致');
    cls = unique(y); C = numel(cls);
    T = zeros(C,n); for i=1:C, T(i, y==cls(i)) = 1; end
end

% 若没有折索引则创建 10 折
if ~(exist('train_idx','var') && exist('test_idx','var'))
    K = 10; yv = vec_from_onehot(T); cv = cvpartition(yv,'KFold',K);
    train_idx = cell(1,K); test_idx = cell(1,K);
    for k=1:K, train_idx{k}=find(training(cv,k)); test_idx{k}=find(test(cv,k)); end
else
    K = numel(train_idx);
end

% 参数（与原 demo 一致）
para.alpha=0.01; para.beta=0.01; para.gamma=10; para.lambda=1; para.mu=1;
para.maxiter=10; para.maxit=5; para.k=10;

rho = 0.10;                              % 候选标签保留比例（可改：0.05/0.10/0.20/...）
ACC=[]; NMI=[];

for it=1:K
    Itr = train_idx{it}; Ite = test_idx{it};

    % 取数据并用训练集统计量标准化
    Xtr = data(Itr,:); Xte = data(Ite,:);
    [Xtr,mu,sigma]=zscore(Xtr); sigma(sigma==0)=1; Xte=(Xte-mu)./sigma;

    % 训练用候选标签
    if exist('partial_target','var')
        P = partial_target(:, Itr);
    else
        P = gen_partial(T(:,Itr), rho);  % 若无 partial_target 则生成
    end
    Yte = T(:, Ite);

    groups = PLC(Xtr, P, Xte, para);
    [acc,nmi] = CalMetrics(Yte, groups);
    ACC(end+1)=acc; NMI(end+1)=nmi;
end

fprintf('ACC: %f std: %f\n', mean(ACC), std(ACC));
fprintf('NMI: %f std: %f\n', mean(NMI), std(NMI));

% ===== 辅助函数（脚本内） =====
function y = vec_from_onehot(T), [~,y]=max(T,[],1); y=y(:); end
function P = gen_partial(T, rho)
    [C,N]=size(T); P=zeros(C,N); m = max(1, ceil(1/rho));  % 候选集合大小
    for i=1:N
        ci = find(T(:,i),1); P(ci,i)=1;
        if m>1
            pool = setdiff(1:C,ci); k = min(m-1, numel(pool));
            if k>0, idx = randperm(numel(pool),k); P(pool(idx),i)=1; end
        end
    end
end
