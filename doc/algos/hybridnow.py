cost_approximator = CostToGo.from_steps(-1)  # Init zero-length (identity) cost-to-go

for step in reversed(range(len(cost_to_go))):  # $t=T, T-1, \dots, 0$
    # Produce Objective for Control Optimization:
    objective = cost_to_go[step] + cost_approximator  # $K_t(s_t, a_t) + \tilde{V}_{t+1}(F_t(s_t, a_t, \xi_{t+1}))$
    
    random_effects_sampler = random_effects_samplers[step]  # $\Xi_{t+1\slice T}$

    # Control Optimization:
    training_state_sampler = training_state_samplers[step]  # $\hat{S}_t$
    params = objective.control_functions[0].parameters()  # $\theta_t$
    optimizer = Optimizer(params)  # Optimizes $A_t$
    for _ in range(control_optimization_gd_steps):
        training_state = training_state_sampler.sample(sample_size)
        rand_effs = random_effects_sampler.sample(sample_size)
        cost = objective(training_state, rand_effs).mean()
    
        cost.backward()
        optimizer.step()  
        optimizer.zero_grad()
    objective.eval()  # Fix implicitly learned parameters
    
    # Produce a Cost Function Approximator:
    cost_approximator = CostToGo.from_steps(0)  # $\tilde{V}_t$
    cost_approximator.cost_functions[0] = V(**nn_config)
    
    # Cost Function Approximation:
    training_state_sampler = approximation_training_state_samplers[step]
    params = cost_approximator.cost_functions[0].parameters()
    optimizer = Optimizer(params)
    for _ in range(approximation_gd_steps):
        training_state = training_state_sampler.sample(sample_size)
        rand_effs = random_effects_sampler.sample(sample_size)
        cost = objective(training_state, rand_effs)  # $V_t(\hat{s}_t)$
        approx_cost = cost_approximator(training_state)  # $\tilde{V}_t(\hat{s}_t)$
        approx_error = torch.norm(cost - approx_cost, p=2)  # $\Vert V_t - \tilde{V}_t\Vert_2$
        
        approx_error.backward()
        optimizer.step()
        optimizer.zero_grad()
    cost_approximator.eval()  # Fix implicitly learned parameters