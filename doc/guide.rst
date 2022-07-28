.. _guide

.. highlight::
    python

.. role:: pycd(code)
    :class: highlight
    :language: python

Guide
=====

Suppose you have an *optimal control problem* at hand which means that: 

* you found a rule to identify a sequence of subsequent, future real-life situations with a sequence of real number vectors $s_0,\dots, s_T$ that, depending on some circumstances $a_0,\dots, a_T$ that you control and some unforeseeable future circumstances $\xi_1,\dots, \xi_T$ (both, again, identified in a pre-defined way with sequences of real number vectors), play out in a mathematically determined way: For a range of *state functions* $F_0,\dots, F_{T-1}$ have $$s_{t+1} = F_t(s_t, a_t, \xi_{t+1}), \quad t=0,\dots, T-1$$

* you are able - when time $t$ will have come and $s_t$ and $a_t$ will be known - to put a cost $K_t(s_t, a_t)$ on the present scenario using a range of *cost functions* $K_0,\dots, K_T$ such that, in retrospect, the total advantageousness of how the problem turned out for you is captured by the *total cost* $$K^{F,a}(s_0,\xi_1,\dots, \xi_T) = \sum_{t=0}^T K_t(s_t, a_t)$$

In this case, one calls $a_0,\dots, a_T$ *controls* and $s_0,\dots, s_T$ the *states* as controlled by $a=(a_0,\dots, a_T)$ under the influence of the *random effects* $\xi_1,\dots, \xi_T$.

.. Moreover, the state functions and control functions are called the *defining functions* of the problem.

Optimal Controls and Cost-To-Go's
---------------------------------

Shifting from the introductory, retrospective formulation to a-priorily introducing the initial state, the random effects, and the choice of controls as random variables $S_0$, $\Xi=(\Xi_1,\dots, \Xi_T)$, and $A=(A_0,\dots, A_T)$, respectively, makes the prospective expectation $EK^{F,A}(S_0,\Xi)$ of the total cost a defined quantity (modulo technical conditions).
This quantity, the *expected total cost* is most sensibly associated with the controls $A$ which in contrast to the other data (including the initial state and the random effects) are "free" variables and the relevant contestants for optimality.
In this sense, a control $\bar{A}=(\bar{A}_0,\dots, \bar{A}_T)$ that among all possible controls $A=(A_0,\dots, A_T)$ minimizes the expected total cost is called an *optimal control* and the infimum value of the expected total cost (that $EK^{F,\bar{A}}(S_0, \Xi)$ meets in this case) is more generally called the *cost-to-go* of the problem.

.. Optimal controls are of importance because, first, they may derive their importance from the cost-to-go they imply which often is of importance when the expectation is taken under a measure of practival relevance (e.g. in risk neutral pricing). will have done so, retrospectively, in every scenario.
.. As a consequence, an accessible form of the optimal control can be of great practical importance as they are guaranteed to perform optimally going forward.

At each time, the conditional expectation of the total cost (conditional on the information accumulated until the time step) is of immediate theoretical and practical importance.
The scope of :py:mod:`ml_adp` is limited to *Markovian* formulations of optimal control problems that feature independent random effects $\Xi_1,\dots, \Xi_T$, which implies the conditional expectations of the total cost to essentially be functions of the current state only.
The assumption of Markovianity is accurate in many practical applications and its presence suggests the feasibility of generic, machine learning based approaches to the numerical solution of optimal control problems:
Since the conditional expectations of the total cost are functions of the respective current state only, optimal controls $\bar{A} = (\bar{A}_0,\dots, \bar{A}_T)$ are guaranteed to be found, at each time $t$, in the form of *control functions* $\tilde{A}_t(s_t)$ in function classes within which common neural network architectures have universal approximation capabilities, meaning that there is a function $\tilde{A}_t(s_t)$ such that $\bar{A}_t = \tilde{A}_t(\bar{S}_t)$, where $\bar{S}$ the state evolution controlled by $\bar{A}$, and that there is a neural network based function $\tilde{A}^{\theta_t}_t(s_t)$ such that $\tilde{A}_t(\bar{S}_t) \approx \tilde{A}_t^{\theta_t}(S_t)$ with high probability.

.. and for which these conditional expectations are easily tractable by virtue of having optimal controls that at each time $t$ are implied to fully factorize over the current state $S_t$, meaning there is a *control function* $s_t\mapsto \tilde{A}_t(s_t)$ for which $\bar{A}_t = \tilde{A}_t(S_t)$ is an optimal control

:py:mod:`ml_adp` serves the implementation of numerical approaches to Markovian optimal control problems motivated by this fact.
It defines the class :class:`ml_adp.cost.CostToGo` which saves the problem data $F=(F_0,\dots, F_{T-1})$ and $K=(K_0,\dots, K_T)$ as well as a particular control $A=(A_0,\dots, A_T)$ in (parametrized) control function form and, as a Python callable, implements the total cost function $(s_0, \xi)\mapsto K^{F, A}(s_0,\xi)$.
As such, it allows the numerical simulation of the effect of $A$ on the total cost in the context of a particular initial state $S_0$ and random effects $\Xi$:
If provided with samples of $S_0$ and $\Xi$, then it computes the corresponding samples of $K^{F, A}(S_0,\Xi)$.
Relying on the automatic differentation capabilites of Pytorch, the user can then differentiate through this numerical simulation and apply gradient descent methods to modify the object-state (the *parameters*) of the control functions $A$ to have $A_t$ turn into the relevant optimal control $\bar{A}_t$ for all times $t$, or, in other words, to transform the :class:`ml_adp.cost.CostToGo`-object into an implementation of the cost-to-go of the problem (as defined above), making apparent the connection of its class to the namesake mathematical object.

:py:class:`ml_adp.cost.CostToGo` exports a list-like interface for its instances that implies effective composition-decomposition properties and facilitates leveraging the so-called *dynamic programming principle* in the above approaches.
The relevant algorithms rely on the approximate satisfaction of the Bellman equations and are the object of study of the field of *approximate dynamic programming*.


Implementing the Optimal Control Problem
----------------------------------------

To elucidate the generic usage of the package and how to leverage it in approximate dynamic programming, assume concretely that implementations of the state functions $F_0,\dots, F_{T-1}$ and the cost functions $K_0,\dots, K_T$ are saved as the entries of Python lists :pycd:`F` and :pycd:`K` of length $4$ and $5$, respectively (implying $T=4$).

Begin by importing :class:`ml_adp.cost.CostToGo` and setting the number of steps between the time points of the problem::

    >>> from ml_adp.cost import CostToGo
    >>> number_of_steps = 4

Create an empty :class:`ml_adp.cost.CostToGo` of appropriate length and inspect it::

    >>> cost_to_go = CostToGo.from_steps(number_of_steps)
    >>> cost_to_go
    CostToGo(
     time |       state_func       |      control_func      |       cost_func        
    ================================================================================
       0                                     None                     None          
       1            None                     None                     None          
       2            None                     None                     None          
       3            None                     None                     None          
       4            None                     None                     None          
      (5)           None                                                            
    )

The above representation shows that :pycd:`cost_to_go` accepts five values, each, for the entries of its :py:attr:`ml_adp.cost.CostToGo.state_functions`, :py:attr:`ml_adp.cost.CostToGo.control_functions`, and :py:attr:`ml_adp.cost.CostToGo.cost_functions` attributes and is to be read as a table, indicating for each step the particular state, control and cost function producing the state, control, and cost belonging to that step, respectively.
Accordingly, the zero-th row remains blank in the ``state_func``-column which indicates ``cost_to_go`` to not accept a function producing the zero-th state $S_0$ (the initial state $S_0$ is provided by the user as a function call argument).

For consistency reasons, a $(T+1)$-th state function $F_T$ for the $T+1$-th step is always included in the specification of :py:class:`ml_adp.cost.CostToGo`'s.
$F_T$ produces the *post-problem scope state* $S_{T+1}$ which - if needed - can serve as the initial state to eventual subsequent optimal control problems and whose inclusion into :class:`ml_adp.cost.CostToGo`'s makes these compose nicely (essentially, by virtue of making :py:attr:`ml_adp.cost.CostToGo.state_functions`, :py:attr:`ml_adp.cost.CostToGo.control_functions` and :py:attr:`ml_adp.cost.CostToGo.cost_functions` lists of equal lengths).
No control is computed and no cost is incurred for the post-problem scope state by ``cost_to_go`` and the user may well omit setting (i.e., set to :pycd:`None`) the final state function if the post-problem scope is irrelevant to his concerns.
He will not need to worry about a potential post-problem scope random effect $\xi_{T+1}$ (to be fed to $F_{T+1}$) when doing so (we will expand on this point later).

Currently, the *defining functions* of ``cost_to_go`` are all set to :pycd:`None`, which, as a value for the state and control function of some step, indicates no state argument and no control argument, respectively, to be passed to the defining functions of the following step and, as a value for cost functions, indicates zero cost to be incurred at the respective steps (whatever the state, control and random effects at that step)::

    >>> cost_to_go()  # No matter the input, produce zero cost
    0

Throughout all of :py:mod:`ml_adp`, it is possible to identify :pycd:`None` with the zero-dimensional vector (be it in Euclidean space or in function space).

Change ``cost_to_go`` from doing nothing by filling it with the state functions and cost functions contained in the lists ``F`` and ``K``::

    >>> cost_to_go.state_functions[:-1] = F
    >>> cost_to_go.cost_functions[:] = K
    >>> cost_to_go
    CostToGo(
     time |       state_func       |      control_func      |       cost_func        
    =================================================================================
        0                                     None                    K0(-)          
        1           F0(-)                     None                    K1(-)          
        2           F1(-)                     None                    K2(-)          
        3           F2(-)                     None                    K3(-)          
        4           F3(-)                     None                    K4(-)          
       (5)           None                                                            
    )

This particular output implies the list ``F`` to have been filled with plain Python functions ``F0``, ``F1``, ``F2``, and ``F3``, in this order (and the analagous statement for ``K``).



Control Functions and Samples
-----------------------------

The ``cost_to_go`` is still incomplete in the sense that, with the control functions all set to :pycd:`None`, it would, internally, not provide any control arguments to the state and cost functions of all steps of the numerical simulation.
With the goal being to provide :pycd:`cost_to_go` with control functions $\bar{A}_0(s_0), \dots, \bar{A}_T(s_t)$ that make it the cost-to-go $EK^{F, \bar{A}}(S_0,\Xi)$ of the optimal control problem, the machine learning approach consists of, first, filling :py:attr:`ml_adp.cost.CostToGo.control_functions` with arbitrary neural networks $A^{\theta_0}_0(s_0),\dots, A^{\theta_T}_T(s_T)$ and, after, relying on gradient-descent algorithms to alter the neural networks' parameters $\theta_0,\dots,\theta_T$ to ultimately have them implement optimal controls.

:py:mod:`ml_adp` facilitates this approach by integrating natively into Pytorch, meaning :class:`ml_adp.cost.CostToGo` is a :class:`torch.nn.Module` that properly registers other user-defined :class:`torch.nn.Module`'s in the defining functions attributes as submodules.
With such being so, the user can readily leverage the Pytorch automatic differentiation and optimization framework to differentiate through the numerical simulation of $EK^{F, A}(S_0,\Xi)$ and optimize the parameters of the control functions.

This statement is conditional on, first, the user making sure the defining functions to be interpreted at runtime to consist out of Pytorch *autograd* operations only and, second, sampling :py:class:`torch.Tensor`'s for the underlying numerical data.
By the duck-typing principle of Python/Pytorch dispatcher, conforming to the first requirement is usually a given if the second requirement is fullfilled.
To elucidate the second requirement, make sense of the dimensions of the state spaces, the control spaces and the random effects spaces (which are the integers $n_t$, $m_t$ and $d_t$ for which $s_t\in\mathbb{R}^{n_t}$, $a_t\in\mathbb{R}^{m_t}$ and $\xi_t\in\mathbb{R}^{p_t}$, respectively).
For the sake of readability, we assume them to not change over the course of the problem and the common values to be saved as the variables ``state_space_size``, ``controls_space_size`` and ``random_effects_space_size``, respectively.
Now, for the purpose of :py:mod:`ml_adp` conforming to the second requirement, a sample of the initial state $S_0$ is arranged to be a :py:class:`torch.Tensor` of size ``(N, state_space_size)`` for some *sample size* integer ``N``.
Analagously, a sample of the random effects $\Xi=(\Xi_1,\dots, \Xi_T)$ is any object behaving like a sequence (of length $T$) of :py:class:`torch.Tensor`'s of sizes ``(N, rand_effs_space_size)``, each, which, in particular, makes a :py:class:`torch.Tensor` of size ``(number_of_steps, N, rand_effs_space_size)`` a such sample (in the present case of the dimensions of the random effects spaces not changing).

We define the controls to be compatible with tensors of such sizes::

    class A(torch.nn.Module):
        def __init__(self, hidden_layer_size):
            super(A, self).__init__()

            sizes = (state_space_size, hidden_layer_size, control_space_size)
            self.fnn = ml_adp.nn.FFN.from_config(sizes)

        def forward(self, state):
            return self.ffn(state)

where for this exposition we stick to basic feed-forward neural networks as provided conveniently by :py:class:`ml_adp.nn.FFN` and expose them in terms of the size of their single hidden layer (we could have used :py:class:`ml_adp.nn.FFN`'s directly as control functions without wrapping them into the class ``A``).

Choose a size for the single hidden layers and set the control functions::

    >>> hidden_layer_size = 20;
    >>> for step in range(len(cost_to_go)):
    ...     cost_to_go.control_functions[step] = A(hidden_layer_size)
    ...
    >>> cost_to_go
    CostToGo(
     time |       state_func       |      control_func      |       cost_func        
    =================================================================================
        0                                   A(train)                  K0(-)          
        1           F0(-)                   A(train)                  K1(-)          
        2           F1(-)                   A(train)                  K2(-)          
        3           F2(-)                   A(train)                  K3(-)          
        4           F3(-)                   A(train)                  K4(-)          
       (5)           None                                                            
    )

In each of the rows, :pycd:`A(train)` indicates that the control function at the corresponding step is an object of the :py:class:`torch.nn.Module`-derived class :class:`A` and that, as a :py:class:`torch.nn.Module`, it contains trainable parameters (which are :class:`torch.nn.Parameter`'s that save :class:`torch.Tensor`'s for which the attribute :pycd:`requires_grad` is :pycd:`True`).


Numerical Simulation of Optimal Control Problems
------------------------------------------------

``cost_to_go`` is now ready to perform the numerical simulation of the optimal control problem:
If ``initial_state`` and ``random_effects`` are samples of $S_0$ and $(\Xi_1, \dots, \Xi_T)$, respectively, then

.. code::

    >>> cost = cost_to_go(initial_state, random_effects);

produces a the corresponding sample of the total cost $K^{F, A}(S_0,\Xi)$:
``cost`` is a :py:class:`torch.Tensor` of size ``(N, 1)`` and is aligned with ``initial_state`` and ``random_effects`` along the first axis. 
If `N` is a large-enough integer, then by the principle of Monte-Carlo simulatio (the *law of large numbers*, more precisely) one can expect

.. code::

    >>> cost.mean();

to produce a number close to $EK^{F, A}(S_0,\Xi)$.

To compute the costs, :py:class:`ml_adp.cost.CostToGo` delegates the task of simulating the *state evolution* $S=(S_t)_{t=0}^{T+1}$ as controlled by $A$ to the callable :py:class:`ml_adp.cost.Propagator` of which an instance it saves in its :py:attr:`ml_adp.cost.CostToGo.propagator` attribute::

    >>> cost_to_go.propagator
    Propagator(
     time |       state_func       |      control_func      
    ========================================================
        0                                   A(train)        
        1           F0(-)                   A(train)        
        2           F1(-)                   A(train)        
        3           F2(-)                   A(train)        
        4           F3(-)                   A(train)        
       (5)           None                                   
    )

``cost_to_go.propagator`` provides ``cost_to_go`` with samples of the state evolution (and, at the same time, the controls computed from the control functions as well as a cleaned-up version of the ``random_effects``-argument) through its :py:meth:`ml_adp.cost.Propagator.propagate`-method::

    >>> states, controls, rand_effs  = cost_to_go.propagator.propagate(initial_state, random_effects)

``states`` and ``controls`` are then lists whose entries are samples of the positions $S_t$ and $A_t$ of the state evolution $S$ and the control $A$.
``rand_effs`` instead is just a cleaned-up version of the sample ``random_effects``.

Note that

.. code::

    >>> len(rand_effs) == number_of_steps + 1  # .. == T+1
    True

which means that ``rand_effs`` has one more entry compared to ``random_effects`` which we intially provided to the :py:class:`ml_adp.cost.Propagator` of ``cost_to_go``.
Indeed,

.. code::

    >>> rand_effs[-1] is None
    True

because, internally, the argument ``random_effects`` has been padded on the right with :pycd:`None`-entries to produce, instead, the list ``rand_effs`` of length equal to the length $T+1$ of ``cost_to_go``, including, compared to ``random_effects`` a zero-dimensional additional post-problem scope random effect $\Xi_{T+1}$, required in the computation of the post-problem scope state $S_{T+1}$.
The :pycd:`None`-value invokes the default behavior of not passing any random effects argument to the state function of the respective step which, for the $(T+1)$-th step, is $F_T$.
As $F_T$ is :pycd:`None`, the computation of the final state $S_{T+1}$ is skipped entirely which makes providing a :pycd:`None`-sample for $\Xi_{T+1}$ not cause any issues.
Accordingly, ``states`` is of length $T+1$ as well and includes a value of :pycd:`None` as its last entry (indicating an irrelevant, zero-dimensional $s_{T+1}$).

As a callable, ``cost_to_go.propagator`` implements the function $$F^A(s_0,\xi) = s_{T+1} = F_T(\dots F_1(F_0(s_0, A_0(s_0), \xi_1), \dots) \dots , A_T(\dots), \xi_{T+1})$$ such that, in the current situation,

.. code-block::

    >>> post_problem_state = cost_to_go.propagator(initial_state, random_effects)

sets ``post_problem_state`` to :pycd:`None`.
This allows to use :py:class:`ml_adp.cost.Propagator`s as state functions. 


Naive Optimization
------------------

.. Pytorch allows to designate a set of :py:class:`torch.Tensor`'s to be tracked in how they enter a particular numerical computation as part of a graph structure and to have their ``.grad``-attribute be populated with the sensitvity of any (intermediate) result of the computation (corresponding to node of the graph) with respect to themselves.
.. As such, Pytorch provides a generic framework for gradient-based optimizatpion methods.

With :py:attr:`ml_adp.cost.CostToGo.control_functions` being a :py:class:`torch.nn.Module`, optimizing all parameters of the control functions of ``cost_to_go`` at once is a trivial, immediately viable approach to control optimzation.

Create a :py:class:`torch.optim.Optimizer` performing the actual gradient descent steps:

.. code-block::
    
    >>> optimizer = torch.optim.SGD(cost_to_go.control_functions.parameters(), lr=learning_rate)

(where ``learning_rate`` is some suitable gradient descent step size) and decide on a number ``N`` for the sample sizes (e.g. 1000) and a number of iterations ``gradient_descent_iterations`` for the gradient descent optimization steps.
After execution of the following code, it is reasonable to expect ``cost_to_go`` to have (more or less) optimal control functions.

.. literalinclude:: ./algos/naive.py

Here, ``initial_state_sampler`` and ``random_effects_sampler`` should produce samples of $S_0$ and $(\Xi_1,\dots, \Xi_T)$, respectively, in terms of the simulation size ``N``.


The Dynamic Programming Principle
---------------------------------


The well-known dynamic programming principle allows to reduce the complexity of the numerical simulation in control function optimization by ensuring that tackling the problem step-wisely produces equivalent results.
Moreover, it promises explicit access to optimal controls to be of practical, scenario-wise applicability.

Defining (subsuming the right technical conditions)
$$V_t(s_t) = \inf_{a_t\in\mathbb{R}^{m_t}} Q_t(s_t, a_t)$$
$$Q_t(s_t, a_t) = K_t(s_t, a_t) + E(V_{t+1}(F_t(s_t, a_t, \Xi_{t+1}))) $$
backwards in time $t$ it is easily seen that $EV_0(S_0)$ constitutes a lower bound for the cost-to-go of the problem and that a control $\bar{A}$ must be optimal if, together with the state evolution $\bar{S}$ it implies, it satisfies
$$\bar{A}_t \in \mathrm{arg\,min}_{a_t\in\mathbb{R}^{m_t}} Q_t(\bar{S}_t, a_t).$$

This principle is known as the *dynamic programming principle* and the equations are called the *Bellman equations*.
They motivate a wide range of numerical algorithms that rely on computing the $V$- and $Q$-functions in a backwards-iterative manner as a first step and determining the optimal controls in a forward pass through the $\mathrm{arg\,min}$-condition as a second step.

In the well-behaved situation (which in particular includes some (semi)-continuity and measurability conditions for the state and cost functions), it can in turn be argued that, subtly, optimal controls $\bar{A}$ (if they exist) turn $EK^{F, \bar{A}}(S_0,\Xi)$ into $EV_0(S_0)$.
This result can be leveraged to formulate algorithms that, using machine learning methods, produce optimal controls as part of the backward pass, eliminating the need for a subsequent forward pass.
To see this, we first make explicit the simple fact that contiguous subcollections of the defining functions again give valid optimal control problems (of a shorter length) for which the above results obviously apply as well:
For all times $t$ we write $\xi_{t, T}$ for $(\xi_t, \dots, \xi_T)$ and $K^{F, A}_{t,T}(s_t, \xi_{t+1, T+1})$ for the total cost function belonging to the optimal control problem given by the state functions $(F_t,\dots, F_T)$, the cost functions $(K_t, \dots, K_T)$, and the control functions $(A_t,\dots, A_T)$.
Using this notation, it can be formulated that a suite of controls $\bar{A}$ is optimal if for all times $t$ $EK_{t,T}^{F, \bar{A}}(\bar{S}_t, \Xi_{t+1, T})$ is minimal when associated with $\bar{A}_t$ (meaning that for all controls $A=(A_0,\dots, A_T)$ for which $A_{t+1}=\bar{A}_{t+1},\dots, A_T=\bar{A}_T$ have $EK_{t,T}^{F, A}(\bar{S}_t, \Xi_{t+1,T})\geq EK_{t,T}^{F, \bar{A}}(\bar{S}_t, \Xi_{t+1,T})$), effectively turning $K_{t-1}(s_{t-1}, a_{t-1}) + E(K_{t,T}^{F, \bar{A}}(F_{t-1}(s_{t-1}, a_{t-1}, \Xi_t), \Xi_{t+1,T}))$ into $Q_{t-1}(s_{t-1}, a_{t-1})$ (which is key).
Some additional mathematical considerations allow to replace $\bar{S}_t$ (which at time $t$ of is not yet available if iterating backwards) in the above by some $\hat{S}_t$ sampled independently from $\Xi_{t+1,T}$ and from a suitable *training distribution* and finally make an explicit backwards-iterative algorithm possible.

:py:class:`ml_adp.cost.CostToGo` implements the :py:class:`ml_adp.cost.CostToGo.__getitem__`-method in a way that makes ``cost_to_go[step:]`` implement $K_{t,T}^{F, A}(s_0,\xi_{t+1, T})$ (if ``step`` corresponds to $t$):

.. autofunction:: ml_adp.cost.CostToGo.__getitem__

For example::

    >>> cost_to_go[3:]
    CostToGo(
     time |       state_func       |      control_func      |       cost_func        
    =================================================================================
        0                                   A(train)                  K3(-)          
        1           F3(-)                   A(train)                  K4(-)          
       (2)           None                                                            
    )

Or::

    >>> cost_to_go[-1]
    CostToGo(
     time |       state_func       |      control_func      |       cost_func        
    =================================================================================
        0                                   A(train)                  K4(-)          
       (1)           None                                                            
    )


In the terms of :py:mod:`ml_adp`, the algorithm as suggested above consists out of stepping backwards through time and optimizing, at each ``step``, the first control function of ``cost_to_go[step:]`` with the objective being ``cost_to_go[step:](state, rand_effs).mean()`` for samples ``state`` and ``rand_effs`` of the training state $\hat{S}_t$ and the random effects $\Xi_{t+1, T}$, respectively:

.. literalinclude:: ./algos/nn_contpi.py

Here, ``training_state_samplers`` and ``random_effects_samplers`` are lists containing the samplers of the relevant random variables (indicated in the code comments).
``Optimizer``, instead, is short for some :py:class:`torch.optim.Optimizer`.

To see the algorithm - which `here`_ has been introduced as the *NNContPi* algorithm - in action, look at the `option hedging example`_.

.. _here: https://arxiv.org/abs/1812.05916v3

.. _option hedging example: examples/option_hedging.html#NNContPi

Value Function Approximation and Hybrid Methods
-----------------------------------------------

The above algorithm has the shortcoming of the numerical simulation getting more and more complex as the backward pass advances.
The technique of *value function approximation* alleviates this issue and, in the present context, consists of replacing, at each step ``step`` in the above algorithm, the tail part of ``objective`` by an (approximately) equivalent other :py:class:`ml_adp.cost.CostToGo` that is computationally more efficient.


.. Entering step ``step``, the ``cost_to_go[step+1]`` assumedly implements $V_{t+1}(s_{t+1}, \xi_{t+1})$ (on the support of the training distribution, that is) such that if some other :py:class:`ml_adp.cost.CostToGo` implements an approximation $\tilde{V}_t(s_t, \xi_t)$ of$EV_t(s_t, \xi_t, \Xi_{t+1, T})$

Mathematically, if for time $t$ some $\tilde{V}_{t+1}(s_{t+1})$ is an approximation of $V_{t+1}(s_{t+1})$, then optimizing $A_t$ in 
$$E(K_t(\hat{S}_t, A_t) + \tilde{V}_{t+1}(F_t(\hat{S}_t, A_t,\Xi_{t+1})))$$
can still be expected to lead to an (approximately) optimal control $\bar{A}_t$ for the original problem.

The composition-decomposition design of :py:class:`ml_adp.cost.CostToGo` makes it easy to perform a such replacement of the time-$(t+1)$ $V$-function ($V$ functions are called *value functions* in the literature where a *value-based* formulation of optimization problems is more common).
In addition to :py:class:`ml_adp.cost.CostToGo` implementing the ``__getitem__``-method as explained in the previous section, it implements the ``__add__``-method:

.. autofunction:: ml_adp.cost.CostToGo.__add__

It does so in a way that makes :py:meth:`ml_adp.cost.CostToGo.__getitem__` and :py:meth:`ml_adp.cost.CostToGo.__add__` work together nicely.
For example::

    >>> cost_to_go[:step] + cost_to_go[step:]  # Split and re-compose
    CostToGo(
     time |       state_func       |      control_func      |       cost_func        
    =================================================================================
        0                                   A(train)                  K0(-)          
        1           F0(-)                   A(train)                  K1(-)          
        2           F1(-)                   A(train)                  K2(-)          
        3           F2(-)                   A(train)                  K3(-)          
        4           F3(-)                   A(train)                  K4(-)          
       (5)           None                                                            
    )

So, in terms of :py:mod:`ml_adp`, it can be formulated that value function approximation consists of replacing, at step ``step``, ``cost_to_go[step:]`` by ``cost_to_go[step] + cost_approximator`` where ``cost_approximator`` is another :py:class:`ml_adp.cost.CostToGo` that approximates ``cost_to_go[step+1:]``.

``cost_approximator`` could be a zero-step :py:mod:`ml_adp.cost.CostToGo`::

    >>> cost_approximator
    CostToGo(
     time |       state_func       |      control_func      |       cost_func        
    =================================================================================
        0                                     None                   V(train)        
       (1)           None                                                            
    )

Here, it is implied that ``cost_approximator.cost_functions[0]`` is neural network based and that it does not require a control argument input.

So, it would, for example, be::

    >>> cost_to_go2 = cost_to_go[0] + cost_approximator
    >>> cost_to_go2
    CostToGo(
     time |       state_func       |      control_func      |       cost_func        
    =================================================================================
        0                                   A(train)                  K0(-)          
        1           F0(-)                     None                   V(train)        
       (2)           None                                                            
    )

and ``cost_to_go2`` would be approximately equal to ``cost_to_go`` if the single control function of ``cost_approximator`` approximated $(s_1, \xi_{2, T}) \mapsto EK_{1,T}^{F, A}(s_1, \xi_{2,T})$ and it would also generally return the same terminal state if one did set ``cost_approximator.state_function[0]`` to ``cost_to_go[1:].propagator`` (or, equivalently, to ``cost_to_go.propagator[1:]`` - :py:class:`ml_adp.cost.Propagator`'s support slicing and concatenation as well)::

    >>> cost_to_go2.state_functions[1] = cost_to_go[1:].propagator
    >>> cost_to_go2
    CostToGo(
     time |       state_func       |      control_func      |       cost_func        
    =================================================================================
        0                                   A(train)                  K0(-)          
        1           F0(-)                     None                   V(train)        
       (2)    Propagator(train)                                                      
    )

In this sense, ``cost_to_go`` and ``cost_to_go2`` are equivalent (as far as approximate solutions are concernced) but ``cost_to_go2`` may well be much more computationally efficient than the multi-step ``cost_to_go``.


The following algorithm introduces value function approximation to the NNContPi algorithm as explained above and `by the same authors`_ has been termed the *HybridNow* algorithm. 

.. _by the same authors: https://arxiv.org/abs/1812.05916v3

.. literalinclude:: ./algos/hybridnow.py

To see this algorithm in action, look, again, at the `option hedging example notebook`_.

.. _option hedging example notebook: examples/option_hedging.html#HybridNow

Choosing the Training Distributions
-----------------------------------

WIP

.. Approximate dynamic programming algorithms require the sampling of states from the so-called training distributions at each time.
..The choice of the training distributions is an empiric process informed by the knowledge of a domain expert.
.. It was argued that reasonable training distributions have their support encompass the support of the distribution of the actual optimal state as estimated by the domain expert.
.. Even if this actual distribution is (partially) discrete (as it is the case for the wealths-part of the state in the multinomial returns model), it makes sense to choose a continuous distribution to better leverage the *generalization capabilities* neural networks.

.. The domain expert may default at each time $t$ to a normal distributions centered at the value that he expects the actual optimal control to take at that time, 

.. It was argued that the training distributions must have support encompassing the supports of the actual distribution of the states and that the more similar the distributions are, the better results one can expect.
.. The domain expert may default to a normal distribution centered at the value he expects to be most relevant in the simulation as a rough proxy for the distributions of the subsequent positions of the state evolution. 


