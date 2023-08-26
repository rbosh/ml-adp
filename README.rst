******
ml-adp
******

``ml_adp`` embeds into Pytorch and serves the numerical solution of Markovian discrete-time stochastic optimal control problems, facilitating the application of user-defined neural networks in control optimization and value function approximation.
Its list-like interfaces make concise implementations of typical backwards-iterative approximate dynamic programming algorithms easy.

Quick Example
-------------

Consider the following state and cost functions together with some number of steps (defining a simple one-dimensional *linear-quadratic* problem):

.. code-block:: python

    steps = 5

    def linear_state(state, control, random_effect):
        return {'state': state + control * (1 + random_effect)}

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
       3  |         None          |         None          |         None         
       4  |         None          |         None          |         None         
       5  |         None          |         None          |         None         
      (6) |         None          |                       |                      
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
       1  |      linear_step      |         None          |     running_cost     
       2  |      linear_step      |         None          |     running_cost     
       3  |      linear_step      |         None          |     running_cost     
       4  |      linear_step      |         None          |     running_cost     
       5  |      linear_step      |         None          |     terminal_cost    
      (6) |         None          |                       |                      
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
       1  |      linear_step      | LinearControl(    ... |     running_cost     
       2  |      linear_step      | LinearControl(    ... |     running_cost     
       3  |      linear_step      | LinearControl(    ... |     running_cost     
       4  |      linear_step      | LinearControl(    ... |     running_cost     
       5  |      linear_step      |         None          |     terminal_cost    
      (6) |         None          |                       |                      
    )

Slice and recompose to perform backward-iterative control optimization and value function approximation and have ``cost_to_go`` turn into the actual cost-to-go function of the given problem:

.. code-block:: python

    >>> objective = cost_to_go[-2:]
    >>> objective
    CostToGo(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       | LinearControl(    ... |     running_cost     
       1  |      linear_step      |         None          |     terminal_cost    
      (2) |         None          |                       |                      
    )
    >>> cost_to_go[:-2] + objective
    CostToGo(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       | LinearControl(    ... |     running_cost     
       1  |      linear_step      | LinearControl(    ... |     running_cost     
       2  |      linear_step      | LinearControl(    ... |     running_cost     
       3  |      linear_step      | LinearControl(    ... |     running_cost     
       4  |      linear_step      | LinearControl(    ... |     running_cost     
       5  |      linear_step      |         None          |     terminal_cost    
      (6) |         None          |                       |                      
    )


Documentation
-------------

Detailed documentation is available `here`__.

__ https://ml-adp.readthedocs.io/en/latest/
