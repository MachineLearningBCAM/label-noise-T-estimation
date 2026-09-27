function main_exp(DATASET,type,N,N_rep,seed,noise_type,noise_p,estimator,method,eps_grid,N_plus,results_path)
    
    data_path = fullfile(fileparts(mfilename('fullpath')), '..', 'data');

    if ischar(N) || isstring(N), N = str2double(N); end
    if ischar(N_rep) || isstring(N_rep), N_rep = str2double(N_rep); end
    if ischar(seed) || isstring(seed), seed = str2double(seed); end
    if ischar(eps_grid) || isstring(eps_grid), eps_grid = str2double(eps_grid); end
    if ischar(N_plus) || isstring(N_plus), N_plus = str2double(N_plus); end

    
    if isempty(eps_grid) || isnan(eps_grid), eps_grid = 0.05; end
    if isempty(N_plus) || isnan(N_plus), N_plus = N/200; end
    
    
    if estimator=="cost"
	    num_workers = str2double(getenv('SLURM_CPUS_PER_TASK'));
	    if ~isnan(num_workers) && num_workers > 1
	        poolobj = gcp('nocreate');
	        if isempty(poolobj)
        	    pc = parcluster('local');
	            pc.NumWorkers = num_workers;
        	    parpool(pc, num_workers);
	        end
	    end
	    disp("pool created");
    end
    
    
    if estimator=="oracle"
        switch DATASET
	            case "MNIST"
    	            conv = load(fullfile(data_path, "mnist_convnext_base.mat"));
   		            num_classes = 10; 	        
                    Features_full = conv.X_train;
                    
                    Features_full = [Features_full;conv.X_test];
                    Features_full = double(Features_full);
                    
                    Y_C_full = double(conv.y_train); % original labels 1-10
                    Y_C_full = [Y_C_full; double(conv.y_test)];
                    Y_C_full(Y_C_full==0) = 10;
    
    
                    X_raw = Features_full;
        	        Y_raw = Y_C_full;
                    X_raw = [X_raw, ones(size(X_raw, 1), 1)];
    
    
            
	            case "letter"
		            load(fullfile(data_path, "letter.mat"))
		            d = 16;
		            num_classes = 26;
		            X_raw = x;
		            Y_raw = y';
                    Y_categorical = categorical(Y_raw);
                    
                    layers = [ ...
                        featureInputLayer(d, 'Name', 'input')
                        fullyConnectedLayer(128, 'Name', 'fc1')
                        reluLayer('Name', 'relu1')
                        
                        fullyConnectedLayer(64, 'Name', 'last_hidden_fc')
                        reluLayer('Name', 'last_hidden_activation') 
                        
                        fullyConnectedLayer(num_classes, 'Name', 'fc_logits')
                        softmaxLayer('Name', 'softmax')
                        classificationLayer('Name', 'output')
                    ];
                    
                   
                    options = trainingOptions('adam', ...
                        'MaxEpochs', 30, ...
                        'MiniBatchSize', 256, ...
                        'InitialLearnRate', 0.001, ...
                        'Shuffle', 'every-epoch', ...
                        'Plots', 'none', ...
                        'Verbose', false);
                   
                    net = trainNetwork(X_raw, Y_categorical, layers, options);
                    
                    
                    X_new = activations(net, X_raw, 'last_hidden_activation', 'OutputAs', 'rows');
                    X_raw = [X_new, ones(size(X_new, 1), 1)];
		            X_raw = double(X_raw);
		            Y_raw = double(Y_raw);
    
    
            
	            case "CIFAR10"
		            load(fullfile(data_path, "cifar10_convnext_base.mat"))
		            d = 1024;
		            num_classes = 10;
		            X_raw = [X_train; X_test];
		            Y_raw = [y_train; y_test];
		            Y_raw(Y_raw==0) = 10;
    
			        X_raw = double(X_raw);
			        Y_raw = double(Y_raw);
    
    
    
	            
	            case "satellite"
		            load(fullfile(data_path, "satellite.mat"))
		            d = 36;
		            num_classes = 6;
		            X_raw = satellite(:,1:36);
		            Y_raw = satellite(:,37);
                    Y_categorical = categorical(Y_raw);
                    
                   
                    layers = [ ...
                        featureInputLayer(d, 'Name', 'input')
                        
                        fullyConnectedLayer(128, 'Name', 'fc1')
                        reluLayer('Name', 'relu1')
                        
                        fullyConnectedLayer(64, 'Name', 'last_hidden_fc')
                        reluLayer('Name', 'last_hidden_activation')
                        
                        fullyConnectedLayer(num_classes, 'Name', 'fc_logits')
                        softmaxLayer('Name', 'softmax')
                        classificationLayer('Name', 'output')
                    ];
                    
                    options = trainingOptions('adam', ...
                        'MaxEpochs', 30, ...
                        'MiniBatchSize', 256, ...
                        'InitialLearnRate', 0.001, ...
                        'Shuffle', 'every-epoch', ...
                        'Plots', 'none', ...
                        'Verbose', false);
                    
                    net = trainNetwork(X_raw, Y_categorical, layers, options);
                    
                    
                    X_new = activations(net, X_raw, 'last_hidden_activation', 'OutputAs', 'rows');
                    X_raw = [X_new, ones(size(X_new, 1), 1)]; 
    
    
		            X_raw = double(X_raw);
		            Y_raw = double(Y_raw);
       
        end
    
    else
        
        switch DATASET
    	        case "MNIST"
        	        load(fullfile(data_path, "mnist.mat"))
        	        d = 784;   
        	        num_classes = 10;
        
        	        X_raw = reshape(training.images,d,[])';
        
        	        X_raw = [X_raw;reshape(test.images,d,[])'];
        
        	        Y_raw = double(training.labels); % original labels 1-10
        	        Y_raw = [Y_raw; double(test.labels)];
        	        Y_raw(Y_raw==0) = 10;
        
	            case "letter"
		            load(fullfile(data_path, "letter.mat"))
		            d = 16;
		            num_classes = 26;
		            X_raw = x;
		            Y_raw = y';
            
	            case "CIFAR10"
		            load(fullfile(data_path, "cifar10_convnext_base.mat"))
		            d = 1024;
		            num_classes = 10;
		            X_raw = [X_train; X_test];
		            Y_raw = [y_train; y_test];
		            Y_raw(Y_raw==0) = 10;
	            
	            case "satellite"
		            load(fullfile(data_path, "satellite.mat"))
		            d = 36;
		            num_classes = 6;
		            X_raw = satellite(:,1:36);
		            Y_raw = satellite(:,37);
        end
    
    end
    
    
    fprintf('Starting %s experiment (%d classes, n=%d)...\n',DATASET, num_classes, N);
    
    
    all_errors = zeros(1,N_rep);
    all_T = zeros(num_classes,num_classes,N_rep);
    T_true = zeros(num_classes,num_classes,N_rep);
    

    for r = 1:N_rep
	    tic
	    rng(seed+r,'philox');
        switch noise_type
            case "uniform"
                if noise_p>1 
	                rho= rand(1,num_classes)/2;
                
	                A = repmat(rho/(num_classes-1), num_classes, 1);
                    A(eye(num_classes)==1) = 1 - rho;
                
                    T_true(:,:,r) = A;
                else
                    T_true(:,:,r) = (1-noise_p) * eye(num_classes) + (noise_p/(num_classes-1)) * (ones(num_classes)-eye(num_classes));
                    T_true(:,:,r) = T_true(:,:,r) ./ sum(T_true(:,:,r), 1);
                end
        
            case "flip"
	            if noise_p>1
                    rho = rand(1,num_classes)/2;
                    T = zeros(num_classes,num_classes);
                    
                    T(1,1) = 1 - rho(1);
                    T(2,1) = rho(1);
                    
                    for j = 2:num_classes
                        T(j,j)   = 1 - rho(j);
                        T(j-1,j) = rho(j);
                    end	 
                    T_true(:,:,r) = T;
                else
                    T_true(:,:,r) =  (1-noise_p)*eye(num_classes) + noise_p*diag(ones(num_classes-1,1),1);
                    T_true(2,1,r) = noise_p;
                
              end
        end
        

        idx = randperm(size(X_raw, 1), N);
        X = X_raw(idx, :);
        Y = Y_raw(idx);
    
    
        Y_tilde = zeros(size(Y));
        for k = 1:N
            probs = T_true(:, Y(k),r);
	    probs = probs(:)';
            Y_tilde(k) = find(rand < cumsum(probs), 1);
        end
    
    
        tic
    
        switch estimator
            case "cost"
                T_estimated = cost(X, Y_tilde, num_classes, eps_grid, N_plus, method);
            case "thr"
                T_estimated = thr(X, Y_tilde, num_classes,N_plus,method);
            case "anchor_based"
                T_estimated = anchor_based(X, Y_tilde, num_classes, method);
	    case "oracle"
		    T_true_ = T_true(:,:,r);
	            T_estimated = oracle(X, Y_tilde, num_classes, T_true_);
        end
    
        err = T_true(:,:,r) - T_estimated;
        all_errors(r) = mean(abs(err(:))); % mean absolute error (MAE)
    
        tiempos(r) = toc; % time
    
        all_T(:,:,r) = T_estimated;
    
        
        fprintf('Repetition %d/%d completed.\n', r, N_rep);
    end
    
    
    switch estimator
        case "cost"
            file_params = [DATASET, string(N), string(N_rep), string(seed), noise_type, string(noise_p), estimator, method, string(eps_grid), string(N_plus)];
        case "thr"
            file_params = [DATASET, string(N), string(N_rep), string(seed), noise_type, string(noise_p), estimator, method, string(N_plus)];
        case "anchor_based"
            file_params = [DATASET, string(N), string(N_rep), string(seed), noise_type, string(noise_p), estimator, method];
        case "oracle"
            file_params = [DATASET, string(N), string(N_rep), string(seed), noise_type, string(noise_p), estimator];
    
    end
    
    
    
    t = datetime("now");
    filename = join(file_params, "_");
    
    
    switch type
	    case "table"
		    out_dir = fullfile(results_path, "table");
		    if ~isfolder(out_dir), mkdir(out_dir); end
		    save(fullfile(out_dir, filename + ".mat"), "tiempos", "all_errors","t", "all_T", "T_true")
	    otherwise
		    out_dir = fullfile(results_path, "plot");
		    if ~isfolder(out_dir), mkdir(out_dir); end
		    disp("saving")
		    save(fullfile(out_dir, type + "_"+ filename + ".mat"),"tiempos", "all_errors","t", "all_T", "T_true")
		    disp("saved")
    end

end


