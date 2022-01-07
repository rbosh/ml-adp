cost_approximator = cost_to_go[-1]  # Init with exact final time cost approximator

for step in reversed(range(len(cost_to_go))):

    sub_ctg = cost_to_go[step] + cost_approximator  # Concatenate
    state_sampler = state_samplers[step]  # Samples  $\hat{S}_t$
    rand_effs_sampler = rand_effs_samplers[step]  # Samples $(\Xi_t,\dots, \Xi_T)$

    # Control Optimization:
    cost_approximator.eval()
    optimizer = Optimizer(sub_ctg.control_functions[0].parameters())  # Optimizes $A_t$
    for _ in range(cost_optimization_gd_steps):
        
        initial_state = state_sampler.sample(N)
        rand_effs = rand_effs_sampler.sample(N)

        cost = sub_ctg(initial_state, rand_effs).mean()
    
        cost.backward()
        optimizer.step()  
        optimizer.zero_grad() 
    
    # Cost Function Approximation:
    sub_ctg.eval()  # Fix implicitly learned parameters (e.g. batch norm mean)
    cost_approximator = next(cost_approximators)
    optimizer = Optimizer(cost_approximator.cost_functions[0].parameters())
    for _ in range(approximation_gd_steps):
        initial_state = state_sampler.sample(approximation_batch_size)
        rand_effs = rand_effs_sampler.sample(N)
       
        with torch.no_grad():  # Training Data for Cost Approximator, No Need for Grad Comps
            cost = torch.tensor([sub_ctg(state, rand_effs) for state in initial_state])
        approx_cost = cost_approximator(initial_state)
        approx_error = torch.norm(cost - approx_cost, p=2)
        
        approx_error.backward()
        optimizer.step()
        optimizer.zero_grad()
    