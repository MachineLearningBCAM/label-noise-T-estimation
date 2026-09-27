function T_mle = oracle(fX, Y_tilde, num_classes, T_true)

    [~, d] = size(fX);
    K = num_classes;
    
    num_params_B = d * K;
    
    
    B0 = ones(d,num_classes);
    
    
    P = T_true;
    P = P./sum(P);
    V0 = log(P);
    
    
    theta0 = [B0(:); V0(:)];

    
    options = optimoptions('fminunc', 'Display', 'none', ...
        'Algorithm', 'quasi-newton', ...
        'SpecifyObjectiveGradient', true, ...
        'MaxIterations', 3000, ...
        'MaxFunctionEvaluations', 1e6);

    nll_func = @(theta) joint_mixture_nll_multiclass(theta, fX, Y_tilde, d, K);
    theta_opt = fminunc(nll_func, theta0, options);

    
    V_opt = reshape(theta_opt(num_params_B + 1 : end), K, K);
    T_mle = exp(V_opt) ./ sum(exp(V_opt), 1);

end

function [nll, grad] = joint_mixture_nll_multiclass(theta, fX, Y_tilde, d, K)
    num_params_B = d * K;
    N = size(fX, 1);
    B = reshape(theta(1:num_params_B), d, K);
    V = reshape(theta(num_params_B + 1 : end), K, K);

    % T(k,j) = P(Y_tilde=k | Y=j)
    V_shift = V - max(V, [], 1); % numerical stability, does not change T
    expV = exp(V_shift);
    T_est = expV ./ sum(expV, 1);

    
    logits = fX * B; % (N x K-1)
    logits = logits - max(logits, [], 2); % numerical stability
    exp_logits = exp(logits); 
    P_clean = exp_logits ./ sum(exp_logits, 2); % (N x K)

    
    P_noisy = P_clean * T_est'; % (N x K)

    P_noisy = max(min(P_noisy, 1 - 1e-9), 1e-9);

    idx = sub2ind(size(P_noisy), (1:N)', Y_tilde);
    observed_probs = P_noisy(idx);

    nll = -sum(log(observed_probs));

    if nargout > 1
        Ty = T_est(Y_tilde, :); % (N x K)

        % dNLL/dP_clean(i,j) = -T(y_i,j) / P_noisy(i,y_i)
        g = -Ty ./ observed_probs; % (N x K)

        
        C = g .* P_clean; % (N x K)
        h = sum(C, 2);    % (N x 1), = sum_j g(i,j) P_clean(i,j)

        
        D = P_clean .* (g - h); % (N x K)
        grad_B = fX' * D;       % (d x K)

        
        Iy = sparse(1:N, Y_tilde, 1, N, K);
        term1 = full(Iy' * C);      % (K x K)
        S = sum(C, 1);              % (1 x K)
        term2 = T_est .* S;         % (K x K)
        grad_V = term1 - term2;     % (K x K)

        grad = [grad_B(:); grad_V(:)];
    end
end
