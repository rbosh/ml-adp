import torch
from ml_adp import CostToGo
from typing import Any

sample_size: int  # Number of samples
gradient_descent_steps: int  # Number of gradient descent iterations

cost_to_go: CostToGo

training_state_sampler: Any  # Sampler for $\hat{S}_t$ in terms of $t$ and sample size
random_effects_sampler: Any  # Sampler for $\Xi_{t \slice T}$ in terms of $t$ and sample size

V: torch.nn.Module  # Neural network for value function approximation
config: dict[str, Any]  # Configuration for neural network


cost_approximator = CostToGo.from_steps(-1)  # Init zero-length (identity) cost-to-go

for time in reversed(range(len(cost_to_go))):  # $t=T, T-1, \dots, 0$
    # Produce Objective for Control Optimization:
    objective = cost_to_go[time] + cost_approximator  # $K_t(s_t, a_t) + \tilde{V}_{t+1}(F_t(s_t, a_t, \xi_{t+1}))$
    
    # Control Optimization:
    optimizer = torch.optim.AdamW(objective.control_functions[0].parameters)  # Optimizes $A_t$
    for _ in range(gradient_descent_steps):
        training_state = training_state_sampler.sample(time, sample_size)  # $\hat{S}_t$
        random_effects = random_effects_sampler.sample(time + 1, sample_size)  # $\Xi_{t+1, T}`
        
        expected_cost = objective(training_state, random_effects).mean()
    
        expected_cost.backward()
        optimizer.step()  
        optimizer.zero_grad()
    objective.eval()
    
    # Produce cost function approximator:
    cost_approximator = CostToGo.from_steps(0)  # $\tilde{V}_t$
    cost_approximator.cost_functions[0] = V(**config)
    
    # Cost function approximation:
    optimizer = torch.optim.AdamW(cost_approximator.cost_functions[0].parameters())
    for _ in range(gradient_descent_steps):
        training_state = training_state_sampler.sample(time, sample_size)
        random_effects = random_effects_sampler.sample(time + 1, sample_size)

        expected_cost = objective(training_state, random_effects)  # $V_t(\hat{S}_t)$
        approx_cost = cost_approximator(training_state)  # $\tilde{V}_t(\hat{S}_t)$
        approximation_error = torch.linalg.norm(expected_cost - approx_cost)  # $\Vert V_t - \tilde{V}_t\Vert_2$
        
        approximation_error.backward()
        optimizer.step()
        optimizer.zero_grad()
    cost_approximator.eval()