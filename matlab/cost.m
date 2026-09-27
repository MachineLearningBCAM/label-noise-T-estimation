function T_hat = cost(X, Y_tilde, num_classes, eps, N_plus, method)

    [n, d] = size(X);

    
    if isempty(eps), eps = 0.05; end
    if isempty(N_plus), N_plus = n/200; end

    
    idx = randperm(n);
    mid = floor(n/2);
    S1_idx = idx(1:mid);
    S2_idx = idx(mid+1:end);

    X1 = X(S1_idx, :); Y1 = Y_tilde(S1_idx);
    X2 = X(S2_idx, :); Y2 = Y_tilde(S2_idx);

    n1 = size(X1, 1);
    m2 = size(X2, 1);
    T_hat = zeros(num_classes, num_classes);
    c_grid = eps:eps:(1-eps);
    Q = numel(c_grid);


    t_tree = [];
    if strcmpi(method, "randomforest")
        t_tree = templateTree('MaxNumSplits', 200, 'MinLeafSize', 5, 'NumPredictorsToSample', round(sqrt(d))); % d is size(X_aug,2)
    end

    task_j = repelem(1:num_classes, Q);
    task_c = repmat(c_grid, 1, num_classes);
    total_tasks = num_classes * Q;

    T_jj_task = -Inf(1, total_tasks);
    coverage_task = false(m2, total_tasks);

    parfor t = 1:total_tasks

        maxNumCompThreads(1);

        j = task_j(t);
        c = task_c(t);

        mask_not_j = (Y1 ~= j);
        X_not_j = X1(mask_not_j, :);
        n_not_j = size(X_not_j, 1);

        X_aug = [X1; X_not_j];
        Y_aug = [ones(n1, 1); -ones(n_not_j, 1)];
        W_aug = [ones(n1, 1) * c; ones(n_not_j, 1)];

        switch method
            case "logistic"
                model = fitclinear(X_aug, Y_aug, 'Weights', W_aug, 'Learner', 'logistic');
            case "svm"
                model = fitclinear(X_aug, Y_aug, 'Weights', W_aug, 'Learner', 'svm');
            case "kernel"
                model = fitcsvm(X_aug, Y_aug, 'Weights', W_aug, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
            case "fitcnet"
               model = fitcnet(X_aug, Y_aug, 'Weights', W_aug, 'LayerSizes', [128, 64], 'Lambda', 0.001);
            case "randomforest"
                model = fitcensemble(X_aug, Y_aug, 'Weights', W_aug, 'Method', 'Bag', 'NumLearningCycles', 300, 'Learners', t_tree);
        end

        % evaluate on S2
        scores2 = predict(model, X2);
        coverage2 = (scores2 == 1); % accepted set where h(x) = +1
        coverage_task(:, t) = coverage2;
        num_accepted = sum(coverage2);

        if num_accepted >= N_plus
            % empirical P(Y_tilde = j | h_c(X) = +1)
            T_jj_task(t) = sum(Y2(coverage2) == j) / num_accepted;
        end
    end

    for j = 1:num_classes
        t_range = (j - 1) * Q + (1:Q);
        [best_T_jj, rel_idx] = max(T_jj_task(t_range));

        if best_T_jj > -Inf
            best_coverage2 = coverage_task(:, t_range(rel_idx));
            coverage_h_best = sum(best_coverage2);
            T_col = zeros(num_classes, 1);
            for i = 1:num_classes
                T_col(i) = sum(Y2(best_coverage2) == i) / coverage_h_best;
            end
        else
            fprintf('Warning: Class %d found no valid c. Filling column uniformly.\n', j);
            T_col = ones(num_classes, 1) / num_classes;
        end

        T_hat(:, j) = T_col;
    end
end
