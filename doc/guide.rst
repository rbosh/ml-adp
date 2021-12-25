.. _guide

.. highlight:: python
.. currentmodule:: ml_adp

Guide
=====

Suppose you have an optimal control problem at hand which means that: 

* you found a rule to identify a sequence of subsequent, prospective real-life situations (a scenario) with a sequence of real number vectors $s_0,\dots, s_T$ that, depending on some circumstances $a_0,\dots, a_T$ that you control and some unforeseeable circumstances $\xi_0,\dots, \xi_T$ (both, again, identified in a pre-defined way with sequences of real number vectors), play out in a mathematically determined way: For a range of *state functions* $F_0,\dots, F_{T-1}$ have $$s_{t+1} = F_t(s_t, a_t, \xi_{t+1}), \quad t=0,\dots, T-1$$

* you are able - when time $t$ will have come and $\xi_t$ will be known - to put a cost $k_t(s_t, a_t, \xi_t)$ on the present situation and circumstances using a range of *cost functions* $k_0,\dots, k_T$ such that, in retrospect, the total advantageousness of how the problem turned out for you is captured by the *total cost* $$k^{F,a}(s_0,\xi_0,\dots, \xi_T) = \sum_{t=0}^T k_t(s_t, a_t, \xi_t)$$

Given a such situation, one calls $a_0,\dots, a_T$ *controls* and $(s_0,\dots, s_T)$ the *states* as controlled by $a=(a_0,\dots, a_T)$ under the influence of the *random effects* $\xi_0,\dots, \xi_T$.
Moreover, the state functions and control functions are called the *defining functions* of the problem.

Optimal Controls and Cost-To-Go's
---------------------------------

Shifting from the introductory retrospective formulation to a-priorily introducing the initial state, the random effects and the choice of controls as random variables $S_0$, $\Xi=(\Xi_0,\dots, \Xi_T)$, and $A=(A_0,\dots, A_T)$, respectively, makes the prospective expectation $Ek^{F,A}(S_0,\Xi)$ of the total cost a defined quantity (modulo technical conditions) and the optimal control problem open to numerical simulation.
This quantity, the *expected total cost* is most sensibly associated with the controls $A$ which in contrast to the initial state and the random effects are "free" variables and contestants for optimality.
In this sense, a control $\bar{A}=(\bar{A}_0,\dots, \bar{A}_T)$ that among all possible controls $A=(A_0,\dots, A_T)$ minimizes the expected total cost is called an *optimal control* and the infimum value that $Ek^{F,\bar{A}}(S_0, \Xi)$ meets in this case is more generally called the *cost-to-go* of the problem.

.. Optimal controls are of importance because, first, they may derive their importance from the cost-to-go they imply which often is of importance when the expectation is taken under a measure of practival relevance (e.g. in risk neutral pricing). will have done so, retrospectively, in every scenario.
.. As a consequence, an accessible form of the optimal control can be of great practical importance as they are guaranteed to perform optimally going forward.

At each time step, the conditional expectation of the cost-to-go (conditional on the information accumulated until the current time step) is of immediate theoretical and practical importance in many applications.
The scope of :py:mod:`ml_adp` is limited to such formulations of optimal control problems for which these conditional expectations are explicitly available by virtue of having optimal controls that at each time $t$ are implied to factorize over the current state $S_t$ and random effect $\Xi_t$ (meaning there is a *control function* $(s_t, \xi_t)\mapsto \tilde{A}_t(s_t, \xi_t)$ for which $\bar{A}_t = \tilde{A}_t(S_t, \Xi_t)$).
The theory shows that, in practice, the scope is effectively not limited by these assumptions and suggests the feasiability to numerical simulation:
In the well-behaved situation and, in particular, if the family of random effects $(\Xi_0,\dots, \Xi_T)$ is independent or if $\Xi_{t-1}$ factorizes over $\Xi_t$ for all times $t$ (meaning $\Xi_t$ includes the informational content of $\Xi_{t-1}$), then optimal controls are guaranteed to be found in the form of *control functions* $A_t(s_t,\xi_t)$ (via $A_t = A_t(S_t, \Xi_t)$ in function classes within which common neural network architectures have universal approximation capabilities.

:py:mod:`ml_adp` serves the implementation of numerical approaches to optimal control problems motivated by this fact.
It defines the class :class:`ml_adp.cost.CostToGo` which saves the problem data $F$ and $k$ as well as a particular control $A$ in control function form and, as callable, implements the total cost function $(s_0, \xi)\mapsto k^{F, A}(s_0,\xi)$.
As such, it allows the numerical simulation of the effect of $A$ on the total cost in the context of a particular initial state $S_0$ and random effects $\Xi$:
If provided with samples of $S_0$ and $\Xi$, then it computes the corresponding samples of $k^{F, A}(S_0,\Xi)$.
Relying on the automatic differentation capabilites of Pytorch, the user can then differentiate through the numerical simulation and apply gradient descent methods to modify the object-state (the *parameters*) of the control functions $A$ to have $A_t$ turn into the relevant optimal control $\bar{A}_t$ for all times $t$, or, in other words, to transform the :class:`ml_adp.cost.CostToGo`-object into the cost-to-go of the problem, making apparent the connection of the class to the namesake mathematical object.

Moreover, :py:mod:`ml_adp` exports a list-like interface to the underlying data $F, A, k$ that implies effective composition-decompositional properties and facilitate leveraging the so-called dynamic programming principle as part of this approach.
The resulting algorithms rely on the approximate satisfaction of the Bellman equations and are the object of study of the field of approximate dynamic programming.


Implementing the Optimal Control Problem
----------------------------------------

To elucidate the generic usage of the package and how to leverage it in approximate dynamic programming, assume concretely that implementations of the state functions $F_0,\dots, F_{T-1}$ and the cost functions $k_0,\dots, k_T$ are saved as the entries of some Python lists :code:`F` and :code:`k` of length $4$ and $5$, respectively (implying $T=4$).

Begin by importing :class:`ml_adp.cost.CostToGo` and setting the number of steps between the time points of the problem::

    >>> from ml_adp.cost import CostToGo
    >>> number_of_steps = 4

Create an empty :class:`ml_adp.cost.CostToGo` of appropriate length and inspect it::

    >>> cost_to_go = CostToGo.from_steps(number_of_steps)
    >>> cost_to_go
    CostToGo(
    step |       state_func       |      control_func      |       cost_func        
    ================================================================================
       0                                     None                     None          
       1            None                     None                     None          
       2            None                     None                     None          
       3            None                     None                     None          
       4            None                     None                     None          
      (5)           None                                                            
    )

The above representation shows that :code:`cost_to_go` accepts five values, each, for the entries of its :py:attr:`ml_adp.cost.CostToGo.state_functions`, :py:attr:`ml_adp.cost.CostToGo.control_functions`, and :py:attr:`ml_adp.cost.CostToGo.cost_functions` attributes and is to be read as a table, indicating for each step (that is, row) the particular state, control and cost function producing the state, control, and cost belonging to that step, respectively.
Accordingly, the zero-th row remains blank in the ``state_func``-column which indicates the cost-to-go to not accept a function producing the zero-th state $s_0$ (the initial state $s_0$ is provided to the numerical simulation by the caller as a function call argument).
For consistency reasons, a $T$-th state function $F_T$ is always included in the specification of :py:class:`ml_adp.cost.CostToGo`'s.
$F_T$ produces the *post-problem scope state* $s_{T+1}$ which - if needed - can serve as the initial state to eventual subsequent optimal control problems and whose inclusion into :class:`ml_adp.cost.CostToGo`'s makes these compose nicely (essentially, by virtue of making :py:attr:`ml_adp.cost.CostToGo.state_functions`, :py:attr:`ml_adp.cost.CostToGo.control_functions` and :py:attr:`ml_adp.cost.CostToGo.cost_functions` lists of equal length).
No control is computed and no cost is incurred for the post-problem scope state by ``cost_to_go`` and the user may well omit setting the final state function if the post-problem scope is irrelevant.

Currently, all of the defining functions are set to :code:`None`, which, as a value for the state and control function of some step, indicates no state argument and no control argument, respectively, to be passed to the defining functions of the following step and, as a value for cost functions, indicates zero cost to be incurred at the respective steps (whatever the state, control and random effects at that step)::

    >>> cost_to_go()  # No matter the input, produce zero cost
    0

Throughout all of :py:mod:`ml_adp`, it is possible to identify `None` with the zero-dimensional vector (be it in Euclidean space or in function spaces).
Change ``cost_to_go`` from doing nothing by filling it with the state functions and cost functions contained in the lists ``F`` and ``k``::

    >>> cost_to_go.state_functions[:-1] = F
    >>> cost_to_go.cost_functions[:] = k
    >>> cost_to_go
    CostToGo(
    step |       state_func       |      control_func      |       cost_func        
    ================================================================================
       0                                     None                    k0(-)          
       1           F0(-)                     None                    k1(-)          
       2           F1(-)                     None                    k2(-)          
       3           F2(-)                     None                    k3(-)          
       4           F3(-)                     None                    k4(-)          
      (5)           None                                                            
)

This particular output implies the list ``F`` to have been filled with plain Python functions ``F0``, ``F1``, ``F2``, and ``F3``, in this order (and the same for ``k``).



Control Functions
-----------------

The ``cost_to_go`` is still incomplete in the sense that, with the control functions all set to ``None``, it would not provide any control argument to the state and cost functions at each step.
With the goal being to provide :code:`cost_to_go` with control functions $\bar{A}_0(s_0, \xi_0), \dots, \bar{A}_T(s_t, \xi_T)$ that make it the cost-to-go $k^{F, \bar{A}}(s_0,\xi)$ of the optimal control problem, the machine learning approach consists of, first, filling :py:attr:`ml_adp.cost.CostToGo.cost_functions` with arbitrary neural networks $A^{\theta_0}_0(s_0,\xi_0),\dots, A^{\theta_T}_T(s_T,\xi_T)$ and, after, relying on gradient-descent algorithms to alter the neural networks' parameters $\theta_0,\dots,\theta_T$ to ultimately have them implement optimal controls.

:py:mod:`ml_adp` facilitates this approach by integrating natively into Pytorch, meaning :class:`ml_adp.cost.CostToGo` to be a :class:`torch.nn.Module` that properly registers other user-defined :class:`torch.nn.Module`'s in the defining functions attributes as submodules.
With such being so, the user can readily leverage the Pytorch automatic differentiation and optimization framework to differentiate through the numerical simulation of $Ek^{F, A}(S_0,\Xi)$ and optimize the parameters of the control functions.
This statement is conditional on, first, the user making sure the defining functions to be interpreted at runtime to consist out of Pytorch-native operations only and, second, sampling :py:class:`torch.Tensor`'s for the underlying numerical data.
By the duck-typing principle of Python/jit of Pytorch, conforming to the first requirement is usually a given if the second requirement is fullfilled.
To elucidate the second requirement, make sense of the dimensions of the state spaces, the control spaces and the random effects spaces (which are the integers $n_t$, $m_t$ and $d_t$ for which $s_t\in\mathbb{R}^{n_t}$, $a_t\in\mathbb{R}^{m_t}$ and $\xi_t\in\mathbb{R}^{d_t}$, respectively).
For the sake of readability, we assume them to not change over the course of the problem and the common values to be saved as the variables ``state_space_size``, ``controls_space_size`` and ``random_effects_space_size``, respectively.
Now, for the purpose of :py:mod:`ml_adp` conforming to the second requirement, a sample of the initial state $S_0$ is a :py:class:`torch.Tensor` of size ``(N, state_space_size)`` for some *sample size* ``N``.
Analagously, a sample for the random effects $\Xi=(\Xi_0,\dots, \Xi_T)$ is any object behaving as a sequence of :py:class:`torch.Tensor`'s of sizes ``(N, rand_effs_space_size)``, each, which, in particular, makes a :py:class:`torch.Tensor` of size ``(number_of_steps+1, N, rand_effs_space_size)`` a such sample.

We define the controls to be compatible with tensors of such sizes::

    class A(torch.nn.Module):
        def __init__(self, hidden_layer_size):
            super(A, self).__init__()

            sizes = (state_space_size + rand_effs_space_size, hidden_layer_size, control_space_size)
            self.fnn = ml_adp.nn.FFN.from_config(sizes)

        def forward(self, state, random_effect):
            return self.ffn(torch.cat([state, random_effect], dim=1))

where for this exposition we stick to plain feed-forward neural networks as provided conveniently by :py:mod:`ml_adp.nn.FFN`and expose them in terms of the size of their single hidden layer.

.. It is an intuitive result that if the cost functions do not depend on the random effects, then an optimal control does, effectively, also not depend on the random effects: $A_t(s_t, \xi_t) = A_t(s_t)$.

.. For illustrative purposes, we assume this to be the case and stick to plain feed-forward neural networks as conveniently provided by :py:mod:`ml_adp.nn.FFN`


    class A(torch.nn.Module):
        def __init__(self, state_space_size, hidden_layer_size, control_space_size):
            super(A, self).__init__()

            sizes = (state_space_size, hidden_layer_size, control_space_size)
            self.fnn = ml_adp.nn.FFN.from_config(sizes)

        def forward(self, state, random_effect):
            return self.ffn(state)

Choose a size for the single hidden layers and set the control functions::

    >>> hidden_layer_size = 20;
    >>> for step in range(len(cost_to_go)):
    ...     cost_to_go.control_functions[step] = A(hidden_layer_size)
    ...
    >>> cost_to_go
    CostToGo(
    step |       state_func       |      control_func      |       cost_func        
    ================================================================================
       0                                   A(train)                  k0(-)          
       1           F0(-)                   A(train)                  k1(-)          
       2           F1(-)                   A(train)                  k2(-)          
       3           F2(-)                   A(train)                  k3(-)          
       4           F3(-)                   A(train)                  k4(-)          
      (5)           None                                                            
    )

In each of the rows, :code:`A(train)` indicates that the control function at the corresponding step is an object of the :py:class:`torch.nn.Module`-derived :class:`A` and that, as a :py:class:`torch.nn.Module`, it contains trainable parameters (which are :class:`torch.nn.Parameter`'s that save :class:`torch.Tensor`'s for which the attribute :code:`requires_grad` is :code:`True`).


Samples and Numerical Simulation of Optimal Control Problems
------------------------------------------------------------


``cost_to_go`` is now ready to perform the numerical simulation of the optimal control problem:
If ``initial_state`` and ``random_effects`` are samples of $S_0$ and $(\Xi_0,\dots, \Xi_T)$, respectively, then

.. code::

    >>> cost = cost_to_go(initial_state, random_effects);

produces a sample ``cost`` of the total cost $k^{F, A}(S_0,\Xi)$:
``cost`` is a :py:class:`torch.Tensor` of size ``(N, 1)`` and if `N` is a large-enough integer, then by the principle of Monte-Carlo simulation one can expect

.. code::

    >>> cost.mean();

to produce a number close to $Ek^{F, A}(S_0,\Xi)$.

To compute the costs, :py:class:`ml_adp.cost.CostToGo` delegates the task of simulating the state evolution $S=(S_t)_{t=0}^T$ as controlled by $A$ to the callable :py:class:`ml_adp.cost.Propagator` of which an instance it saves in its :py:attr:`ml_adp.cost.CostToGo.propagator` attribute::

    >>> cost_to_go.propagator
    Propagator(
    step |            state_func             |           control_func            
    =============================================================================
       0                                                   A(train)              
       1                 F0(-)                             A(train)              
       2                 F1(-)                             A(train)              
       3                 F2(-)                             A(train)              
       4                 F3(-)                             A(train)              
      (5)                None                                                    
    )

``cost_to_go.propagator`` provides ``cost_to_go`` with samples of the state evolution (and, at the same time, the controls computed from the control functions as well as a cleaned-up version of the ``random_effects``-argument) through its :py:meth:`ml_adp.cost.Propagator.propagate`-method::

    >>> states, controls, rand_effs  = cost_to_go.propagator.propagate(initial_state, random_effects)

``states`` and ``controls`` are then lists whose entries are samples of the positions $S_t$ and $A_t$ of the state evolution $S$ and the control $A$.
``rand_effs`` instead is just a cleaned-up version of the sample ``random_effects``.

Note::

    >>> len(rand_effs) == number_of_steps + 2
    True

This means that ``rand_effs`` has one more entry compared to ``random_effects``.
Indeed,

.. code::

    >>> rand_effs[-1] is None
    True

because, internally, the argument ``random_effects`` has been padded on the right with a :code:`None`-entry to produce, instead, the list ``rand_effs`` of length equal the number of random effects actually required (which includes $\Xi_{T+1}$ for the computation of the terminal state $S_{T+1}$).
The :code:`None`-value invokes the default behavior of not passing any random effects argument to the defining functions of the $(T+1)$-th step (which is $F_T$, only).
As $F_T$ is ``None``, the computation of the final state $S_{T+1}$ is skipped entirely which makes not providing a sample for $\Xi_{T+1}$ not cause any issues.
Accordingly, ``states`` is of length $T+2$ as well and includes a value of ``None`` as its last entry.

As a callable, ``cost_to_go.propagator`` implements the function $$F^A(s_0,\xi) = s_{T+1} = F_T(\dots F_1(F_0(s_0, A_0(s_0,\xi_0), \xi_1), \dots) \dots , A_T(\dots, \xi_T), \xi_{T+1})$$ such that, in the current situation,

.. code-block::

    >>> post_problem_state = cost_to_go.propagator(initial_state, random_effects)

sets ``post_problem_state`` to :code:`None`.


Naive Optimization
------------------

Pytorch allows to designate a set of :py:class:`torch.Tensor`'s to be tracked in how they enter a particular numerical computation as part of a graph structure and to have their ``.grad``-attribute be populated with the sensitvity of any (intermediate) result of the computation (corresponding to node of the graph) with respect to themselves.
As such, Pytorch provides a generic framework for gradient-based optimization methods.

With :py:mod:`ml_adp` being native to Pytorch, taking all parameters of the control functions of ``cost_to_go`` and and performing gradient-descent using the gradients collected with respect to the numerical approximation of the final cost as computed above presents itself as an immediately viable approach to control-optimzation.

Create a :py:mod:`torch.optim.Optimizer` performing the actual gradient descent steps:

.. code-block::
    
    >>> optimizer = torch.optim.SGD(cost_to_go.control_functions.parameters(), lr=learning_rate)

and decide on a number ``N`` for the sample sizes and a number of iterations ``I`` for the gradient descent optimization steps.
After execution of the following code it is reasonable to expect the above to have modified the parameters of the control functions as for the control functions to be (more or less) optimal.

.. literalinclude:: ./naive.py

Here, ``initial_state_samplers`` and ``random_effects_sampler`` should produce samples of $S_0$ and $(\Xi_0,\dots, \Xi_T)$, respectively.


The Dynamic Programming Principle
---------------------------------


The well-known result of dynamic programming allows to reduce the complexity of the optimization and to tackle the step-wisely and ensures the optimal controls to work scenario-wisely.
Defining
$$V_t(s_t, \xi_t) = \inf_{a_t\in\mathbb{R}^{m_t}} Q_t(s_t, a_t, \xi_t)$$
$$Q_t(s_t, a_t, \xi_t) = k_t(s_t, a_t, \xi_t) + E(V_{t+1}(F_t(s_t, a_t, \Xi_{t+1}))) $$
it is easily seen that $EV_0(S_0,\Xi)$ constitutes a lower bound to the cost-to-go of the problem and that a control $\bar{A}$ must be optimal if together with the state evolution $\bar{S}$ it implies it satisfies
$$\bar{A}_t \in \mathrm{arg\,min}_{a_t\in\mathbb{R}^{m_t}} Q_t(\bar{S}_t, a_t, \Xi_t).$$
This principle is known as the *dynamic programming principle* and the equations are called the Bellman equations and it motivates a wide range of numerical algorithms that rely on computing the $V$- and $Q$-functions in a backwards-iterative manner as a first step and computing the optimal controls by a forward pass through the $\mathrm{arg\,min}$-condition as a second step.

In the well-behaved situation (which in particular entails some (semi)-continuity and measurability conditions), it can in turn be argued that, subtly, optimal controls $\bar{A}$ turn $Ek^{F, \bar{A}}(S_0,\Xi)$ into $EV_0(S_0,\Xi)$.
This result can be leveraged using machine learning methods to formulate algorithms that produce optimal controls as part of the backward pass, eliminating the need for a subsequent forward pass.
To see this, we first make explicit the simple fact that contiguous subcollections of the defining functions again give valid optimal control problems (of a shorter length) for which the above results obviously apply as well:
For all times $t$ we write $\xi_{t, T}$ for $(\xi_t, \dots, \xi_T)$ and $k^{F, A}_{t,T}(s_t, \xi_{t, T})$ for the total cost function belonging to the optimal control problem given by the state functions $(F_t,\dots, F_T)$, the cost functions $(k_t, \dots, k_T)$, and the control functions $(A_t,\dots, A_T)$.
Using this notation, it can be formulated that a suite of controls $\bar{A}$ is optimal if for all times $t$ $Ek^{F, \bar{A})(\bar{S}_t, \Xi_{t,T})$ is minimal among all varying the time-$t$ control.
turning $k_t(s_t, a_t, \xi_t) + E(k_{t+1}^{F, \bar{A}}(F_t(s_t, a_t, \Xi_{t+1}), \Xi_{t+1,T}))$ into $Q_t(s_t, a_t, \xi_t)$.
Some additional mathematical considerations allow to replace $\bar{S}_t$ (which at time $t$ of the backwards iteration is not yet available) in the above by some $\hat{S}_t$ sampled independently from $\Xi_{t,T}$ if done so from a suitable *training distribution* which finally makes an explicit backwards-iterative algorithm possible.

:py:class:`ml_adp.cost.CostToGo` implements the :py:class:`ml_adp.cost.CostToGo.__getitem__`-method in a way that makes ``cost_to_go[step:]`` implement $k^{F, A}(s_0,\xi_{t, T})$ (if ``step`` corresponds to $t$).
For example::

    >>> cost_to_go[3:]
    CostToGo(
    step |       state_func       |      control_func      |       cost_func        
    ================================================================================
       0                                   A(train)                  k3(-)          
       1           F3(-)                   A(train)                  k4(-)          
      (2)           None                                                            
    )


.. currentmodule:: ml_adp.cost

.. autofunction:: CostToGo.__getitem__

In the terms of :py:mod:`ml_adp` the algorithm as suggested above consists out of stepping backwards through time and optimizing, at each step ``step``, the first control function of ``cost_to_go[step:]`` with the objective being ``cost_to_go[step:](state, rand_effs).mean()`` for samples ``state`` and ``rand_effs`` of the training state $\hat{S}_t$ and the random effects $\Xi_{t, T}$, respectively.
Which allows a very concise implementation

.. literalinclude:: ./policyiteration.py

Here, ``state_samplers`` and ``rand_effs_samplers`` are lists containing the relevant samplers as explained in the comments.

To see the algorithm - which `here`_ has been termed the *NNContPi* algorithm - in action, look at the `option hedging example`_.

.. _here: https://arxiv.org/abs/1812.05916v3

.. _option hedging example: examples/option_hedging.html#NNContPi

Value Function Approximation and Hybrid Methods
-----------------------------------------------

The above algorithm has the shortcoming of the numerical simulation getting more complex as the number of steps in the backward pass decrement.
The technique of *value function approximation* alleviates this issue and, in the present context, consists of replacing, at each step ``step`` in the above algorithm, the ``sub_ctg[step+1:]`` part of ``cost_to_go[step:]`` by an (approximately) equivalent other :py:class:`ml_adp.cost.CostToGo` that is computationally more efficient:
Entering step ``step``, the ``cost_to_go[step+1]`` assumedly implements $V_t(s_t, \xi_t)$ (on the support of the training distribution, that is) such that if some other :py:class:`ml_adp.cost.CostToGo` implements an approximation $\tilde{V}_t(s_t, \xi_t)$ of$EV_t(s_t, \xi_t, \Xi_{t+1, T})$

Mathematically, if at step $t$ $\tilde{V}_{t+1}(s_{t+1}, \xi_{t+1})$ is an approximation of $V_{t+1}(s_{t+1}, \xi_{t+1})$, then optimizing $A_t$ in 
$$E(k_t(\hat{S}_t, A_t) + \tilde{V}_{t+1}(F_t(\hat{S}_t, A_t,\Xi_{t+1}), \Xi_{t+1}))$$
(in ... stead as part of the above algorithm) can still be expected to lead to an (approximately) optimal control $\bar{A}_t$ for the exact problem.

The composition-decomposition design of :py:mod:`ml_adp` makes it easy to compose as in the above equation.
In addition to :py:class:`ml_adp.cost.CostToGo` implementing the ``__getitem__`` method as explained above, it implements the ``__add__`` method in a way the makes it compatible with decomposition::

    >>> cost_to_go[:step] + cost_to_go[step:]
    CostToGo(
    step |       state_func       |      control_func      |       cost_func        
    ================================================================================
       0                                   A(train)                  k0(-)          
       1           F0(-)                   A(train)                  k1(-)          
       2           F1(-)                   A(train)                  k2(-)          
       3           F2(-)                   A(train)                  k3(-)          
       4           F3(-)                   A(train)                  k4(-)          
      (5)           None                                                            
    )

So, in terms of :py:mod:`ml_adp`, value function approximation consists of replacing, at step ``step``, ``cost_to_go[step:]`` by ``cost_to_go[step] + cost_approximator`` where ``cost_approximator`` is another :py:class:`ml_adp.cost.CostToGo` that approximates ``cost_to_go[step+1:]``.
E.g., ``cost_approximator`` could be a zero-step :py:mod:`ml_adp.cost.CostToGo`::

    >>> cost_approximator
    CostToGo(
    step |       state_func       |      control_func      |       cost_func        
    ================================================================================
       0                                     None                NNCost(train)      
      (1)           None                                                            
    )

Here, it is indicated that ``cost_approximator.cost_functions[0]`` is neural-network based and that it does not require a control argument input.

So, it would, for example, be::

    >>> cost_to_go[0] + cost_approximator
    CostToGo(
    step |       state_func       |      control_func      |       cost_func        
    ================================================================================
       0                                   A(train)                  k0(-)          
       1           F0(-)                     None                NNCost(train)      
      (2)           None                                                            
    )

which would be approximately equal to ``cost_to_go`` if the single control function of ``cost_approximator`` approximated $(s_1, \xi_{1, T}) \mapsto Ek_{1,T}^{F, A}(s_1, \xi_{1,T)$ (which would also generally return the same terminal state if one sets ``cost_approximator.state_function[0] = cost_to_go[1:].propagator`` beforehand).


.. literalinclude:: ./hybrid.py

To see this algorithm - which `here`_ has been termed the *HybrdigNow* algorithm - in action, look, again, at the `option hedging example notebook`_.

.. _here: https://arxiv.org/abs/1812.05916v3

.. _option hedging example notebook: examples/option_hedging.html#HybridNow

Choosing the Training Distributions
-----------------------------------

Approximate dynamic programming algorithms as explained in :ref:`_quick_guide` require the sampling of states from the so-called training distributions at each time.
The choice of the training distributions is an empiric process informed by the knowledge of a domain expert.
It was argued that reasonable training distributions have their support encompass the support of the distribution of the actual optimal state as estimated by the domain expert.
Even if this actual distribution is (partially) discrete (as it is the case for the wealths-part of the state in the multinomial returns model), it makes sense to choose a continuous distribution to better leverage the *generalization capabilities* neural networks.

The domain expert may default at eacth time to a normal distributions centered at the value that he expects the actual optimal control to take at that time, 

It was argued that the training distributions must have support encompassing the supports of the actual distribution of the states and that the more similar the distributions are, the better results one can expect.
The domain expert may default to a normal distribution centered at the value he expects to be most relevant in the simulation as a rough proxy for the distributions of the subsequent positions of the state evolution. 
He will also think about the range of the prices he expects over the course of the market steps in order to make an informate decision on the variance of this training distribution.
For the wealth process, he may use a similar procedure:


