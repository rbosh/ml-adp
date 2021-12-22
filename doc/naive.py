for i in range(I):
    initial_state = initial_state_sampler.sample((N, state_space_size))
    rand_effs = rand_effs_sampler.sample((len(cost_to_go), N, rand_effs_space_size))

    cost = cost_to_go(initial_state, rand_effs).mean()
  
    cost.backward()  # Compute the Gradients
    optimizer.step()  # Perform a Gradient Descent Step
    optimizer.zero_grad()  # Ready the Next Step
