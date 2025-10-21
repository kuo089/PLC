% -------------------------------------------------------------------------
% PLC Demo for MSRCv2 - All Rho Values with Parameter Tuning
% -------------------------------------------------------------------------
clear; clc; close all;

% --- 1. Load Data ---
fprintf('Loading MSRCv2 dataset...\n');
load('MSRCv2_Sample.mat');

% --- 2. Set Parameters ---
% Parameter ranges to test, as specified in the paper
para_alpha_values = [0.01, 0.1, 1];
para_beta_values = [0.01, 0.1, 1];
para_k_values = [10, 15, 20, 25, 30, 40];

% Fixed parameters
para.gamma = 10;
para.lambda = 1;
para.mu = 1;
para.maxiter = 10; % Iterations for AdversarialPCP
para.maxit = 5;    % Iterations for the main PLC loop

% Proportions (rho) to be tested
rho_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]; 
num_iterations = 10; % Number of random partitions for each setting

% --- 3. Main Experiment Loop ---
num_samples = size(data, 1);

% Store the best results for each rho
best_results = table('Size', [length(rho_values), 6], ...
                     'VariableTypes', {'double', 'double', 'double', 'double', 'double', 'double'}, ...
                     'VariableNames', {'rho', 'best_alpha', 'best_beta', 'best_k', 'best_ACC', 'best_NMI'});

% Loop for each rho value
for r_idx = 1:length(rho_values)
    rho = rho_values(r_idx);
    fprintf('\n--------------------------------------------------\n');
    fprintf('Running experiments for rho = %.2f\n', rho);
    fprintf('--------------------------------------------------\n');
    
    best_avg_acc = 0;
    best_params = struct();

    % --- Loop through all parameter combinations ---
    for alpha = para_alpha_values
        for beta = para_beta_values
            for k = para_k_values
                
                para.alpha = alpha;
                para.beta = beta;
                para.k = k;
                
                fprintf('  Testing params: alpha=%.2f, beta=%.2f, k=%d ...\n', alpha, beta, k);
                
                current_ACC = [];
                current_NMI = [];

                % --- Inner loop for 10 random partitions ---
                for it = 1:num_iterations
                    % Create random partitions
                    num_train = round(num_samples * rho);
                    shuffled_indices = randperm(num_samples);
                    train_idx_dynamic = shuffled_indices(1:num_train);
                    test_idx_dynamic = shuffled_indices(num_train+1:end);

                    % Prepare data
                    train_data = zscore(data(train_idx_dynamic, :));
                    test_data = zscore(data(test_idx_dynamic, :));
                    train_p_target = partial_target(:, train_idx_dynamic);
                    test_target = target(:, test_idx_dynamic);

                    % Run PLC
                    try
                        groups = PLC(train_data, train_p_target, test_data, para);
                        [acc, nmi] = CalMetrics(test_target, groups);
                        current_ACC = [current_ACC, acc];
                        current_NMI = [current_NMI, nmi];
                    catch ME
                        fprintf('    Error during PLC execution: %s\n', ME.message);
                        % Assign a very low value to penalize this parameter set
                        current_ACC = [current_ACC, 0]; 
                        current_NMI = [current_NMI, 0];
                    end
                end
                
                % Check if this parameter set is the best so far for the current rho
                avg_acc_for_set = mean(current_ACC);
                if avg_acc_for_set > best_avg_acc
                    best_avg_acc = avg_acc_for_set;
                    best_params.alpha = alpha;
                    best_params.beta = beta;
                    best_params.k = k;
                    best_params.avg_acc = avg_acc_for_set;
                    best_params.std_acc = std(current_ACC);
                    best_params.avg_nmi = mean(current_NMI);
                    best_params.std_nmi = std(current_NMI);
                end
            end
        end
    end
    
    % --- Store and display the best result for the current rho ---
    best_results.rho(r_idx) = rho;
    best_results.best_alpha(r_idx) = best_params.alpha;
    best_results.best_beta(r_idx) = best_params.beta;
    best_results.best_k(r_idx) = best_params.k;
    best_results.best_ACC(r_idx) = best_params.avg_acc;
    best_results.best_NMI(r_idx) = best_params.avg_nmi;
    
    fprintf('\n  Best result for rho = %.2f (averaged over %d runs):\n', rho, num_iterations);
    fprintf('  Best Params -> alpha: %.2f, beta: %.2f, k: %d\n', best_params.alpha, best_params.beta, best_params.k);
    fprintf('  ACC: %f std: %f\n', best_params.avg_acc, best_params.std_acc);
    fprintf('  NMI: %f std: %f\n', best_params.avg_nmi, best_params.std_nmi);
end

fprintf('\n\n==================== FINAL SUMMARY ====================\n');
disp(best_results);
fprintf('======================================================\n');