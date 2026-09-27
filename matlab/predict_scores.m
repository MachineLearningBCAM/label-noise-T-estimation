function scores = predict_scores(model, X_input)
    [~, scores] = predict(model, X_input);
end
