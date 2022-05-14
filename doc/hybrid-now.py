cost_approximator = cost_to_go[-1]  # Init with exact final time cost approximator

for step in reversed(range(len(cost_to_go) - 1)):  # $t=T-1, T-2, \dots, 0$
    # Produce Objective for Control Optimization:
    objective = cost_to_go[step] + cost_approximator  # $K_t^{F_t,A_t} \concat \tilde{V}_{t+1}$
    
    training_state_sampler = training_state_samplers[step]  # $\hat{S}_t$
    random_effects_sampler = random_effects_samplers[step]  # $\Xi_{t\slice T}$

    # Control Optimization:
    params = objective.control_functions[0].parameters()  # $\theta_t$
    optimizer = Optimizer(params)  # Optimizes $A_t$
    for _ in range(control_optimization_gd_steps):
        
        training_state = training_state_sampler.sample(N)
        rand_effs = random_effects_sampler.sample(N)

        cost = objective(training_state, rand_effs).mean()
    
        cost.backward()
        optimizer.step()  
        optimizer.zero_grad()
    objective.eval()  # Fix implicitly learned parameters
    
    # Produce a Cost Function Approximator:
    cost_approximator = CostToGo.from_steps(0)  # $\tilde{V}_t$
    cost_approximator.cost_functions[0] = V(**nn_config)
    
    # Cost Function Approximation:
    params = cost_approximator.cost_functions[0].parameters()
    optimizer = Optimizer(params)
    for _ in range(approximation_gd_steps):
        
        training_state = state_sampler.sample(approximation_batch_size)
        rand_effs = random_effects_sampler.sample(N)
       
        with torch.no_grad():  # Produce training data for cost approximator, no need for grad comps
            cost = torch.stack([objective(state).mean(dim=0)
                                for state in training_state],
                               dim=0)  # $V_t(\hat{s}_t)$
        
        approx_cost = cost_approximator(training_state)  # $\tilde{V}_t(\hat{s}_t)$
        approx_error = torch.norm(cost - approx_cost, p=2)  # $\Vert V_t - \tilde{V}_t\Vert_2$
        
        approx_error.backward()
        optimizer.step()
        optimizer.zero_grad()
    cost_approximator.eval()  # Fix implicitly learned parameters
    