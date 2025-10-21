%% Main script for Dual-Graph Regularized Adaptive Multi-view Feature Selection (DGR-AMFS)
% =========================================================================
% Description:
% This script implements the DGR-AMFS algorithm.
%
% Author: Wang Kuo (Concept) & Gemini (Implementation)
% Date: 2025-10-21
% =========================================================================

clear;
clc;
close all;

% Add necessary paths
addpath(genpath('core'));
addpath(genpath('utils'));
addpath(genpath('./')); % Assumes PLC code is in the current path

%% 1. Data Preparation
fprintf('1. Loading MSRCv2 Data...\n');
load('MSRCv2_Sample.mat'); 

% Adapt data format for the new algorithm
Y = partial_target'; % partial_target is c x n, we need n x c
[~, true_labels_indices] = max(target, [], 1);
true_labels = true_labels_indices'; % n x 1 ground truth
[n, c] = size(Y);

% DGR-AMFS is a multi-view algorithm.
% The provided MSRCv2_Sample is single-view.
% We simulate two views by adding a small amount of noise for demonstration.
% In your actual research, please load true multi-view data here.
X = cell(1, 2);
X{1} = zscore(data); % View 1: Original data, standardized
noise = 0.1 * randn(size(data));
X{2} = zscore(data + noise); % View 2: Data with noise, standardized
V = length(X);

fprintf('Data prepared. %d samples, %d views, %d classes.\n\n', n, V, c);

%% 2. Set Hyperparameters
fprintf('2. Setting Hyperparameters...\n');
params.V = V;
params.c = c;

% Weights for objective function terms
params.lambda_s = 1e-3;   % For L2,1 sparsity
params.beta = 1e-2;       % For Graph 1 (feature manifold)
params.lambda_a = 1e-1;   % For A-B alignment consistency
params.lambda_B = 1e-2;   % For Graph 2 (consensus graph)
params.lambda_e = 1e-2;   % For consistency-driven view entropy
params.gamma = 2;         % Exponent for the entropy term

% Training process parameters
params.max_iter = 30;     % Max iterations for the main loop
params.warmup_iter = 5;   % Warm-up iterations
params.r_G1 = 3;          % Frequency to reconstruct Graph 1
params.m_PLC = 5;         % Frequency to refresh PLC pseudo-labels (Q)

% Graph construction parameters
params.k_G1 = 10;         % kNN for Graph 1
params.k_G2 = 10;         % kNN for Graph 2
params.sigma = 1.0;       % Bandwidth for Gaussian kernel

% Parameters for the PLC algorithm
% [!! 修复 !!] 将 'plc_para' 重命名为 'plc' 以匹配 PLC_wrapper.m 期望的字段名称
plc_params_struct.alpha = 0.01;
plc_params_struct.beta = 0.01;
plc_params_struct.gamma = 10;
plc_params_struct.lambda = 1;
plc_params_struct.mu = 1;
plc_params_struct.maxiter = 10; % Iterations for AdversarialPCP
plc_params_struct.maxit = 5;    % Iterations for the main PLC loop
plc_params_struct.k = 10;       % kNN for graph inside PLC
params.plc = plc_params_struct; % Assign the struct to the main params

fprintf('Hyperparameters are set.\n\n');

%% 3. Run DGR-AMFS Algorithm
fprintf('3. Starting DGR-AMFS model training...\n');
[F, U, alpha, obj_values] = DGR_AMFS_train(X, Y, params);

%% 4. Analyze Results
fprintf('\nTraining finished.\n\n');
fprintf('Final view weights alpha:\n');
disp(alpha');

% Calculate accuracy of the final pseudo-labels against ground truth
[~, pred_labels] = max(F, [], 2);

% Use bestMap for label alignment to calculate clustering accuracy (ACC)
% This is a standard practice for evaluating clustering performance.
final_acc = length(find(true_labels == bestMap(true_labels, pred_labels))) / n;
fprintf('Final Accuracy of F (ACC after Hungarian matching): %.4f\n', final_acc);

% Plot the objective function value curve
figure;
plot(1:length(obj_values), obj_values, '-o', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Objective Function Value');
title('DGR-AMFS Objective Function Convergence');
grid on;
set(gca, 'FontSize', 12);

