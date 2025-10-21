zai%load("ecoli r=2_Sample.mat")
%load('lost.mat');

para.alpha=0.01;
para.beta=0.01;
para.gamma=10;
para.lambda=1;
para.mu=1;
para.maxiter=10;
para.maxit=5;
para.k=10; 


predictLabels=[];
ACC=[];
NMI=[]; 

for it=1:10

        % 假设 data1, data2 分别是视图1/视图2的全数据 (1758×d1 / 1758×d2)
    train_data_cell = { data1(train_idx{it},:), data2(train_idx{it},:) };
    test_data_cell  = { data1(test_idx{it},:),  data2(test_idx{it},:)  };

    [groups_all_aligned, perms_all] = PLC_mv(train_data_cell, train_p_target, test_data_cell, para);

    % 取对齐后的各视图“测试集部分”标签，用来评测
    nb = numel(test_idx{it});
    groups_v1 = groups_all_aligned(end-nb+1:end, 1);
    groups_v2 = groups_all_aligned(end-nb+1:end, 2);

    % 你可以任选一个视图做评测，或做简单融合（投票/参考）：
    groups = groups_v1;                      % 例：用对齐后的视图1
    % 或：groups = mode([groups_v1, groups_v2], 2);   % 多视图投票
    [acc,nmi] = CalMetrics(test_target, groups);

    ACC=[ACC,acc];
    NMI=[NMI,nmi];
    
end

fprintf('ACC: %f std: %f\n',mean(ACC),std(ACC));
fprintf('NMI: %f std: %f\n',mean(NMI),std(NMI));


