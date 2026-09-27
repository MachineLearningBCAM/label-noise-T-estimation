function T_tilde = anchor_based(X, Y_tilde, num_classes, method)
    
    switch method
        case "fitcnet"
            model = fitcnet(X, Y_tilde,'LayerSizes',[128,64],'Lambda',0.001);
        case "logistic"
            model = fitcnet(X, Y_tilde,'LayerSizes',[],'Lambda',0.001);
        case "randomforest"
	        t = templateTree('MaxNumSplits', 200, 'MinLeafSize', 5, 'NumPredictorsToSample', sqrt(size(X,2)));
            model = fitcensemble(X, Y_tilde, 'Method', 'Bag', 'NumLearningCycles', 300, 'Learners', t );
    end
    
    T_tilde = zeros(num_classes);
    [~,probs] = predict(model,X);
    
    for j = 1:num_classes 
        [~,I] = sort(probs(:,j));
        a = I(ceil(length(I)*0.97));
        T_tilde(:,j) = probs(a,:)';
    end
end

