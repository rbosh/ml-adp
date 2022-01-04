for step in reversed(range(len(cost_to_go))):
    
    sub_ctg = cost_to_go[step:]  # Implements $k^{F, A}_{t,T}(s_t, \xi_t, \dots, \xi_T)$
    optimizer = torch.optim.SGD(sub_ctg.control_functions[0].parameters())  # Optimizes $A_t$
    state_sampler = state_samplers[step]  # Samples $\hat{S}_t$
    rand_effs_sampler = rand_effs_samplers[step]  # Samples $(\Xi_t, \dots, \Xi_T)$

    for _ in range(gradient_descent_steps):
        
        initial_state = state_sampler.sample((N, state_space_size))
        rand_effs = rand_effs_sampler.sample((len(cost_to_go), N, rand_effs_space_size))

        cost = sub_ctg(initial_state, rand_effs).mean()
    
        cost.backward()  # Compute the Gradients
        optimizer.step()  # Perform a Gradient Descent Step
        optimizer.zero_grad()  # Ready the Next Step