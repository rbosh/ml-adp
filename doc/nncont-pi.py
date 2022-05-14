for step in reversed(range(len(cost_to_go))):
    
    # Produce Objective for Control Optimization:
    objective = cost_to_go[step:]  # $K_{t\slice T}^{F, A}$, in expectation equal to $Q_t(\cdot, A_t(\cdot))$

    training_state_sampler = training_state_samplers[step]  # $\hat{s}_t$
    random_effects_sampler = random_effects_samplers[step]  # $\Xi_{t\slice T}$
        
    # Control Optimization:
    params = objective.control_functions[0].parameters()  # $\theta_t$
    optimizer = torch.optim.SGD(params)  # Optimizes $A_t$
    for _ in range(gradient_descent_steps):
        
        training_state = training_state_sampler.sample(N)  # Sample $\hat{s}_t$
        random_effects = random_effects_sampler.sample(N)  # Sample $\Xi_{t\slice T}$

        cost = objective(training_state, random_effects).mean()  # $EQ_t(\hat{s}_t, A_t(\hat{s}_t))$
    
        cost.backward()  # Compute the Gradients
        optimizer.step()  # Perform a Gradient Descent Step
        optimizer.zero_grad()  # Ready the Next Step
    objective.eval()  # Fix implicitly optimized parameters of $\bar{A}_t,\dots, \bar{A}_T$