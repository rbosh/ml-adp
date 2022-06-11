for step in reversed(range(len(cost_to_go))):  # $t=T, T-1, \dots, 0$
    # Produce Objective for Control Optimization:
    objective = cost_to_go[step:]  # $K_{t\slice T}^{F, A}$, in expectation equal to $Q_t(\cdot, A_t(\cdot))$
    
    training_state_sampler = training_state_samplers[step]  # $\hat{s}_t$
    random_effects_sampler = random_effects_samplers[step]  # $\Xi_{t+1\slice T}$
    
    # Control Optimization:
    params = objective.control_functions[0].parameters()  # $\theta_t$
    optimizer = Optimizer(params)
    for _ in range(gradient_descent_steps):  # Gradient descent steps
        training_state = training_state_sampler.sample(sample_size)  # Sample $\hat{s}_t$
        random_effects = random_effects_sampler.sample(sample_size)  # Sample $\Xi_{t+1\slice T}$
        cost = objective(training_state, random_effects).mean()  # $EQ_t(\hat{s}_t, A_t(\hat{s}_t))$
    
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
    objective.eval()  # Fix implicitly optimized parameters of $\bar{A}_t,\dots, \bar{A}_T$
    
