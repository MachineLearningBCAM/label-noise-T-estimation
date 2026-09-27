function T_hat = thr(X, Y_tilde, num_classes, N_plus, method)

    [n, ~] = size(X);

    
    if isempty(N_plus), N_plus = n/200; end

    
    idx = randperm(n);
    mid = floor(n/2);
    S1_idx = idx(1:mid);
    S2_idx = idx(mid+1:end);
    
    X1 = X(S1_idx, :); Y1 = Y_tilde(S1_idx);
    X2 = X(S2_idx, :); Y2 = Y_tilde(S2_idx);

    T_hat = zeros(num_classes);

    parfor j = 1:num_classes
        
        switch method
            case "logistic"
                model = fitclinear(X1, j*(Y1==j),'Learner', 'logistic');
            case "fitcnet"
                model = fitcnet(X1,j*(Y1==j),'LayerSizes',[128,64],'Lambda',0.001);
	        case "randomforest"
                t = templateTree('MaxNumSplits', 200, 'MinLeafSize', 5, 'NumPredictorsToSample', sqrt(size(X1,2)) );
                model = fitcensemble(X1, j*(Y1==j), 'Method', 'Bag', 'NumLearningCycles', 300, 'Learners', t ); 
        end


	    scores_T = predict_scores(model, X2);
        
        sj_T = scores_T(:, 2);

        [~, idx] = sort(sj_T, 'descend');
        sorted_labels = Y2(idx);

        proportion_j = - Inf;
        for k = floor(N_plus):length(sorted_labels)
            if proportion_j <= sum(sorted_labels(1:k)==j)/k
                selected_labels = sorted_labels(1:k);
                proportion_j = sum(sorted_labels(1:k)==j)/k;
            end
        end

        for k = 1:num_classes
                T_hat(k, j) = sum(selected_labels == k) / length(selected_labels);
        end
    end
end

