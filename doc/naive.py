for _ in range(gradient_descent_iterations):
    initial_state = initial_state_sampler.sample(N)  # Sample of $S_0$
    rand_effs = rand_effs_sampler.sample(N)  # Sample of $(\Xi_0,\dots, \Xi_T)$

    cost = cost_to_go(initial_state, rand_effs).mean()
  
    cost.backward()  # Compute the Gradients
    optimizer.step()  # Perform a Gradient Descent Step
    optimizer.zero_grad()  # Ready the Next Step
