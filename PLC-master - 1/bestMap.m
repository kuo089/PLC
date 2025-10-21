function [newL2] = bestMap(L1,L2)
%bestmap: permute labels of L2 to match L1 as good as possible
%   [newL2] = bestMap(L1,L2);
%
%   version 2.0 --May/2007
%   version 1.0 --November/2003
%
%   Written by Deng Cai (dengcai AT gmail.com)


%===========    

L1 = L1(:);
L2 = L2(:);
if size(L1) ~= size(L2)
    error('size(L1) must == size(L2)');
end

Label1 = unique(L1);
nClass1 = length(Label1);
Label2 = unique(L2);
nClass2 = length(Label2);

nClass = max(nClass1,nClass2);
G = zeros(nClass);
for i=1:nClass1
	for j=1:nClass2
		G(i,j) = length(find(L1 == Label1(i) & L2 == Label2(j)));
	end
end

[c,t] = hungarian(-G);
newL2 = zeros(size(L2));
% 获取 L1 和 L2 中实际存在的唯一标签数
Label1 = unique(L1);
nClass1 = length(Label1);
Label2 = unique(L2);
nClass2 = length(Label2);

% 构建代价矩阵 G (这里假设 Label1 和 Label2 中的标签是从 1 开始的整数)
% 如果标签不是从1开始的连续整数，需要更复杂的处理来正确构建G
nClass = max(nClass1, nClass2);
G = zeros(nClass);
% --- (确保 G 的构建逻辑是正确的，原始代码是正确的) ---
for i=1:nClass1
    for j=1:nClass2
        G(Label1(i), Label2(j)) = length(find(L1 == Label1(i) & L2 == Label2(j))); % 使用标签值作为索引可能不安全，最好用映射
    end
end
% 更安全的构建 G 的方式 (假设标签值可能不连续或非从1开始):
map1 = containers.Map(Label1, 1:nClass1);
map2 = containers.Map(Label2, 1:nClass2);
costMatrix = zeros(nClass1, nClass2);
for k = 1:length(L1)
    if isKey(map1, L1(k)) && isKey(map2, L2(k))
        idx1 = map1(L1(k));
        idx2 = map2(L2(k));
        costMatrix(idx1, idx2) = costMatrix(idx1, idx2) + 1;
    end
end
% 使用匈牙利算法 (需要保证输入是方阵，且越大越好，所以用负值)
costMatrixPad = -padarray(costMatrix, [max(0, nClass2-nClass1), max(0, nClass1-nClass2)], 0, 'post');
[c, ~] = hungarian(costMatrixPad); % c 的长度是 max(nClass1, nClass2)


% 进行映射赋值，增加检查
for i = 1:nClass2
    mapped_label_index = c(i); % 这是 Label1 中的索引 (1 到 nClass1) 或 填充的索引
    if mapped_label_index > 0 && mapped_label_index <= nClass1 % 确保索引有效且在 Label1 的范围内
        newL2(L2 == Label2(i)) = Label1(mapped_label_index);
    else
        % 处理无法映射或映射到填充索引的情况
        % 可以赋一个特殊值，例如 NaN，或者赋一个最常见的 Label1 值
        % 这里暂时赋 NaN，表示该聚类无法映射到测试集中的真实标签
        newL2(L2 == Label2(i)) = NaN;
        % 或者赋第一个真实标签: newL2(L2 == Label2(i)) = Label1(1);
    end
end