******
ml-adp
******

``ml_adp`` embeds into Pytorch and serves the numerical solution of Markovian discrete-time stochastic optimal control problems, facilitating the application of user-defined neural networks in control optimization and value function approximation.
Its list-like interfaces make concise implementations of typical backwards-iterative approximate dynamic programming algorithms easy.

Quick Example
-------------

Consider the following state and cost functions together with some number of steps (defining a simple one-dimensional *linear-quadratic* problem):

.. code-block:: python

    steps = 2

    def linear_state(state, control, z):
        return {'state': state + control * (1 + z)}

    def running_cost(state, control):
        return state ** 2 + control ** 2

    def terminal_cost(state):
        return state ** 2

To solve this problem, create an empty ``CostToGo``-instance of required length:

.. code-block:: python

    >>> from ml_adp import CostToGo
    >>> cost_to_go = CostToGo.from_steps(steps)
    >>> cost_to_go
    CostToGo(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       |         None          |         None         
       1  |         None          |         None          |         None         
       2  |         None          |         None          |         None         
      (3) |         None          |                       |                      
    )

Set the state and cost functions:

.. code-block:: python

    >>> cost_to_go.state_functions[:-1] = linear_step
    >>> cost_to_go.cost_functions[:-1] = running_cost
    >>> cost_to_go.cost_functions[-1] = terminal_cost
    >>> cost_to_go
    CostToGo(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       |         None          |     running_cost     
       1  |     linear_state      |         None          |     running_cost     
       2  |     linear_state      |         None          |     terminal_cost    
      (3) |         None          |                       |                      
    )

Define a parametrized control function architecture:

.. code-block:: python

    import torch

    class LinearControl(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1, bias=False)

        def forward(self, state):
            return {'control': self.linear(state)}

Set the control functions:

.. code-block:: python

    >>> for i in range(len(cost_to_go) - 1):  # No need for control at final time
    ...     cost_to_go.control_functions[i] = LinearControl()
    ...
    >>> cost_to_go
    CostToGo(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       | LinearControl(    ... |     running_cost     
       1  |     linear_state      | LinearControl(    ... |     running_cost     
       2  |     linear_state      |         None          |     terminal_cost    
      (3) |         None          |                       |                      
    )

Make sense of an initial state for the problem and sample a random effect for each step of the simulation:

.. code-block:: python

    >>> initial_state = {'state': torch.tensor([[1.]])}
    >>> random_effects = [{'z': torch.randn(10000, 1)} for _ in range(cost_to_go.steps())]

Simulate the total cost of the problem as incurred by the current control functions:

.. code-block:: python

    >>> cost_to_go(initial_state, random_effects).mean()
    tensor(6.6254, grad_fn=<MeanBackward0>)

Slice and recompose:

.. code-block:: python

    >>> head, tail = cost_to_go[:1], cost_to_go[1:]
    >>> head
    CostToGo(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       | LinearControl(    ... |     running_cost     
      (1) |     linear_state      |                       |                      
    )
    >>> tail
    CostToGo(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       | LinearControl(    ... |     running_cost     
       1  |     linear_state      |         None          |     terminal_cost    
      (2) |         None          |                       |                      
    )
    >>> head + tail
    CostToGo(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       | LinearControl(    ... |     running_cost     
       1  |     linear_state      | LinearControl(    ... |     running_cost     
       2  |     linear_state      |         None          |     terminal_cost    
      (3) |         None          |                       |                      
    )


Slicing and composition is consistent with the functional behavior of ``CostToGo``'s:

.. code-block:: python

    >>> head_cost = head(initial_state, random_effects[:1])
    >>> intermediate_state = head.state_evolution(initial_state, random_effects[:1])
    >>> tail_cost = tail(intermediate_state, random_effects[1:])
    >>> (head_cost + tail_cost).mean()  # Expect the same result as above:
    tensor(6.6254, grad_fn=<MeanBackward0>)

Leverage these properties in the concise formulation of backward-iterative control optimization and value function approximation algorithms and turn ``cost_to_go`` turn into the actual `cost-to-go function`_ of the given problem.

.. _cost-to-go function: https://en.wikipedia.org/wiki/Value_function

Documentation
-------------

Detailed documentation is available `here`__.

__ https://ml-adp.readthedocs.io/en/latest/
