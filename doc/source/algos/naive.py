for _ in range(gradient_descent_steps):
    initial_state = initial_state_sampler.sample(sample_size)  # Sample of $S_0$
    rand_effs = rand_effs_sampler.sample(sample_size)  # Sample of $(\Xi_0,\dots, \Xi_T)$

    expected_cost = cost_to_go(initial_state, rand_effs).mean()
  
    expected_cost.backward()
    optimizer.step()
    optimizer.zero_grad()
