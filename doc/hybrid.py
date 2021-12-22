cost_approximator = cost_to_go[-1]

for step in reversed(range(len(cost_to_go) - 1)):

    sub_ctg = cost_to_go[step] + cost_approximator  # Implements $k^{F, A}_{t,T}(s_t, \xi_t, \dots, \xi_T)$
    initial_state_sampler = state_samplers[step]  # Samples Ŝ from training distribution
    rand_effs_sampler = rand_effs_samplers[step]  # Samples random effects (ξ₀,ξ₁,...)

    # Control Optimization:
    cost_approximator.eval()
    optimizer = Optimizer(sub_ctg.control_functions[0].parameters())  # Optimizes $A_t$
    for i in range(cost_optimization_gd_steps):
       
        initial_state = initial_state_sampler.sample((N, state_space_size))
        rand_effs = rand_effs_sampler.sample((len(cost_to_go), N, rand_effs_space_size))

        cost = sub_ctg(initial_state, rand_effs).mean()
    
        cost.backward()  # Compute the Gradients
        optimizer.step()  # Perform a Gradient Descent Step
        optimizer.zero_grad()  # Ready the Next Step
    
    # Cost Function Approximation:
    sub_ctg.eval()  # Fix implicitly learned parameters (e.g. batch norm mean)
    cost_approximator = next(cost_approximators)
    optimizer = Optimizer(cost_approximator.cost_functions[0].parameters())
    for i in range(approximation_gd_steps):
        initial_state = initial_state_sampler.sample(approximation_batch_size)
        rand_effs = rand_effs_sampler.sample(len(sub_ctg), sims_size)
       
        with torch.no_grad(): 
            cost = torch.tensor([sub_ctg(state, rand_effs) for state in initial_state])
        approx_cost = cost_approximator(initial_state)
        
        approx_error = torch.norm(cost - approx_cost, p=2)
        
        approx_error.backward()
        optimizer.step()
        optimizer.zero_grad()
    