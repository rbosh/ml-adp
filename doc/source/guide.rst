.. _guide:

.. highlight::
    python

.. role:: pycd(code)
    :class: highlight
    :language: python

Guide
=====

Suppose you have an *optimal control problem* at hand which means that 

* you found a rule to identify a number of subsequent, future real-life situations with a sequence $s_0, \dots, s_T$ of data that, depending on some circumstances $a_0, \dots, a_T$ that you control and some unforeseeable circumstances $\xi_1, \dots, \xi_T$ that you don't control, play out in a mathematically determined way: For a range of *state functions* $F_0, \dots, F_{T-1}$ have $$s_{t+1} = F_t(s_t, a_t, \xi_{t+1}), \quad t=0, \dots, T-1,$$

* you are able - when time $t$ will have come and $s_t$ and $a_t$ will be known - to put a cost $K_t(s_t, a_t)$ on the present scenario using a range of *cost functions* $K_0, \dots, K_T$ such that, in retrospect, the total advantageousness of how the problem turned out for you is captured by the *total cost* $$K^{F,a}(s_0, \xi_1, \dots, \xi_T) = \sum_{t=0}^T K_t(s_t, a_t).$$

In this case, one calls $a_0, \dots, a_T$ *controls* and $s_0, \dots, s_T$ the *states* as controlled by $a=(a_0, \dots, a_T)$ under the influence of the *random effects* $\xi_1, \dots, \xi_T$.


Optimal Controls and Cost-To-Go's
---------------------------------

Shifting from the introductory, retrospective formulation to a-priorily introducing the initial state, the random effects, and the choice of controls as random variables $S_0$, $\Xi=(\Xi_1, \dots, \Xi_T)$, and $A = (A_0, \dots, A_T)$, respectively, makes the prospective expectation $EK^{F, A}(S_0, \Xi)$ of the total cost a defined quantity.
This quantity, the *expected total cost*, is associated with the chosen controls, inducing a notion of optimality for controls:
A control $\bar{A} = (\bar{A}_0, \dots, \bar{A}_T)$ that among all possible controls $A = (A_0, \dots, A_T)$ minimizes the expected total cost is called an *optimal control*.
More generally, the infimum value of the expected total cost (that $EK^{F, \bar{A}}(S_0, \Xi)$ meets in the case of optimal controls) is more generally called the *cost-to-go* of the problem.

.. Optimal controls are of importance because, first, they may derive their importance from the cost-to-go they imply which often is of importance when the expectation is taken under a measure of practival relevance (e.g. in risk neutral pricing). will have done so, retrospectively, in every scenario.
.. As a consequence, an accessible form of the optimal control can be of great practical importance as they are guaranteed to perform optimally going forward.

At each time, the conditional expectation of the total cost (conditional on the information accumulated until that time) is of immediate theoretical and practical importance.
The scope of :py:mod:`ml_adp` is limited to *Markovian* formulations of optimal control problems that feature independent random effects $\Xi_1, \dots, \Xi_T$, which at each time implies the conditional expectation of the total cost to essentially be a function of the current state only.
The assumption of Markovianity is expedient in many practical applications and its presence suggests the feasibility of generic, machine learning based approaches to the numerical solution of optimal control problems:
Since the conditional expectations of the total cost are functions of the respective current state only, the time-$t$ optimal control $\bar{A}_t$ is guaranteed to be found in the form of a *control function* (or *feedback function*) $\tilde{A}_t(s_t)$ in function classes within which common neural network architectures have universal approximation capabilities.
This means that there is a function $\tilde{A}_t(s_t)$ such that $\bar{A}_t = \tilde{A}_t(\bar{S}_t)$ (where $\bar{S}$ is the state evolution controlled by $\bar{A}$) and that there is a neural network $\tilde{A}^{\theta_t}_t(s_t)$ such that $\tilde{A}_t(\bar{S}_t) \approx \tilde{A}_t^{\theta_t}(S_t)$ with high probability.

.. and for which these conditional expectations are easily tractable by virtue of having optimal controls that at each time $t$ are implied to fully factorize over the current state $S_t$, meaning there is a *control function* $s_t\mapsto \tilde{A}_t(s_t)$ for which $\bar{A}_t = \tilde{A}_t(S_t)$ is an optimal control

:py:mod:`ml_adp` serves the implementation of numerical methods motivated by this fact.
It defines the class :class:`~ml_adp.base.CostAccumulation`, which saves the problem data $F = (F_0, \dots, F_{T-1})$ and $K = (K_0, \dots, K_T)$ as well as a particular control $A = (A_0, \dots, A_T)$ in control function form and, as a Python callable, implements the total cost function $$(s_0, \xi)\mapsto K^{F, A}(s_0, \xi).$$

As such, it allows the numerical simulation of the effect of $A$ on the total cost in the context of a particular initial state $S_0$ and random effects $\Xi$:
More precisely:
If provided with samples of $S_0$ and $\Xi$, then it computes the corresponding samples of $K^{F, A}(S_0, \Xi)$.
Relying on the automatic differentation capabilites of Pytorch, the user can then differentiate through this numerical simulation and apply gradient descent methods to modify the object-state (the *parameters*) of the control functions $A$, turning $A$ into the relevant optimal control $\bar{A}$, or, in other words, transforming the :class:`~ml_adp.base.CostAccumulation`-object into an implementation of the cost-to-go function of the problem, making apparent the connection of the class to the namesake mathematical object.

:py:class:`~ml_adp.base.CostAccumulation` exports a list-like interface that implies effective composition-decomposition properties and facilitates leveraging the so-called *dynamic programming principle* during control optimization.
The relevant algorithms rely on the approximate satisfaction of the *Bellman equations* and are summarized using the term *approximate dynamic programming*.



Implementing the Optimal Control Problem
----------------------------------------


In :mod:`ml_adp`, the state, control, and random effect at each time are Python dictionaries whose string-type keys more precisely identify the components of the data and whose values are the actual numerical data in the form of Pytorch tensors (or simple floats)::

    >>> import torch

For example, assume that

* the state $s_t$ is given by two numerical components $s_t^{(1)}$ and $s_t^{(2)}$ of sizes $n_1$ and $n_2$,

* $a_t$ is just a single component of size $m$,

* $\xi_{t+1}$ decomposes into two components $\xi_{t+1}^{(1)}$ and $\xi_{t+1}^{(2)}$ of sizes $p_1$ and $p_2$.

Then, the following definitions produce valid state, control, and random effect instances::

    >>> state = {
    ...     'state_1': torch.rand(n_1),
    ...     'state_2': torch.rand(n_2),
    ... }
    ...
    >>> control = {'control': torch.rand(m)}
    >>> random_effect = {
    ...     'random_effect_1': torch.randn(p_1),
    ...     'random_effect_2': torch.randn(p_2),
    ... }
    ...

The signature of the state functions must then be given by (a subset of) the keys of ``state``, ``control``, and ``random_effect``, while its return value must be a dictionary with the state keys.
The cost functions internally are called only with the state and control dictionaries and return their numerical output directly (not wrapped into any dictionary).
For example, the definition of a valid state function (acting as the identity function on the state) reads

.. code-block::

    >>> def identity_state_func(state_1, state_2, control, random_effect_1, random_effect_2):
    ...     return {"state_1": state_1, "state_2": state_2}
    ...

while the definition of a valid cost function (producing zero cost) could read

    >>> def zero_cost_func(state_1, state_2, control):
    ...     return 0.0

Going forward, we assume concretely that such implementations of the state functions $F_0, \dots, F_{T-1}$ and the cost functions $K_0, \dots, K_T$ are saved as the entries of Python lists ``state_functions`` and ``cost_functions`` of length $4$ and $5$, respectively (implying $T=4$).

As a first step, begin by importing :class:`~ml_adp.base.CostAccumulation` and setting the number of steps between the time points $0, \dots, T$ of the problem::

    >>> from ml_adp import CostAccumulation
    >>> steps = 4

Create an empty :class:`~ml_adp.base.CostAccumulation` of appropriate length and inspect it::

    >>> cost_acc = CostAccumulation.from_steps(steps)
    >>> cost_acc
    CostAccumulation(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       |         None          |         None         
       1  |         None          |         None          |         None         
       2  |         None          |         None          |         None         
       3  |         None          |         None          |         None         
       4  |         None          |         None          |         None         
      (5) |         None          |                       |                      
    )

The above representation shows that :pycd:`cost_acc` accepts five values, each, for the entries of its :py:attr:`~ml_adp.base.CostAccumulation.state_functions`-, :py:attr:`~ml_adp.base.CostAccumulation.control_functions`-, and :py:attr:`~ml_adp.base.CostAccumulation.cost_functions`-attributes and is to be read as a table, indicating for each time the particular function producing the state, control, and cost of that time.
Accordingly, the time-$0$ row remains blank in the state function column which is in line with the initial state $S_0$ being provided by the user as a function call argument (and not being produced by a state function).

For consistency reasons, a time-$(T + 1)$ state function $F_T$ is always included in the specification of :py:class:`~ml_adp.base.CostAccumulation`'s.
$F_T$ produces what could be called the *post-problem state* $S_{T+1}$.
The post-problem state can serve as the initial state for eventual subsequent optimal control problems and the inclusion of its state function into :class:`~ml_adp.base.CostAccumulation`'s makes these compose nicely (essentially, by virtue of making their attributes :py:attr:`~ml_adp.base.CostAccumulation.state_functions`, :py:attr:`~ml_adp.base.CostAccumulation.control_functions` and :py:attr:`~ml_adp.base.CostAccumulation.cost_functions` lists of equal lengths).
No control is computed and no cost is incurred for the post-problem scope state by ``cost_acc`` and the user may well omit setting the final state function (i.e., set to :pycd:`None`) if the post-problem state is irrelevant to their concerns.
They will not need to worry about a potential post-problem random effect $\xi_{T+1}$ (to be fed to $F_{T+1}$) when doing so.

Currently, the *defining functions* of ``cost_acc`` are all set to :pycd:`None`, which, as a value for the state and control function of some time, indicates no state argument and no control argument, respectively, to be passed to the defining functions of the following time and, as a value for cost functions, indicates zero cost to be incurred at the respective times (whatever the state, control and random effects at that time)::

    >>> cost_acc()  # No matter the input, produce zero cost
    0.0

Change ``cost_acc`` from doing nothing by filling it with the state functions and cost functions contained in the lists ``state_functions`` and ``cost_functions`` (notice the array-like way of addressing the defining functions containers)::

    >>> cost_acc.state_functions[:-1] = state_functions
    >>> cost_acc.cost_functions[:] = cost_functions
    >>> cost_acc
    CostAccumulation(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       |         None          |      cost_func_0     
       1  |     state_func_1      |         None          |      cost_func_1     
       2  |     state_func_2      |         None          |      cost_func_2     
       3  |     state_func_3      |         None          |      cost_func_3     
       4  |     state_func_4      |         None          |      cost_func_4     
      (5) |         None          |                       |                      
    )

This particular output implies the list ``state_functions`` to have been filled with plain Python functions ``state_func_1``, ``state_func_2``, ``state_func_3``, and ``state_func_4``, in this order (and the analagous statement for ``cost_functions``).



Neural Network Based Control Functions
--------------------------------------

The ``cost_acc`` is still incomplete in the sense that, with the control functions all set to :pycd:`None`, it would, internally, not produce any control values from the state evolution (the user, could provide them as part of the initial state and the random effects arguments, however).
With the goal being to provide :pycd:`cost_acc` with control functions $\bar{A}_0(s_0), \dots, \bar{A}_T(s_t)$ that make it the cost-to-go $EK^{F, \bar{A}}(S_0, \Xi)$ of the optimal control problem, the machine learning approach consists of, first, filling :py:attr:`~ml_adp.base.CostAccumulation.control_functions` with suitable neural networks $A^{\theta_0}_0(s_0), \dots, A^{\theta_T}_T(s_T)$ and, after, relying on gradient-descent algorithms to alter the neural networks' parameters $\theta_0, \dots, \theta_T$ to ultimately have them implement optimal controls.

:py:mod:`ml_adp` facilitates this approach by integrating natively with Pytorch, meaning concretely that :class:`~ml_adp.base.CostAccumulation` is a Pytorch module that properly registers other user-defined Pytorch modules in the defining functions attributes as submodules.
With such being so, the user can readily leverage the Pytorch automatic differentiation and optimization framework to differentiate through the numerical simulation of $EK^{F, A}(S_0, \Xi)$ and optimize the parameters of the control functions (this statement is conditional, of course, on, first, the user making sure the defining functions to be interpreted at runtime to consist out of Pytorch *autograd* operations only and, second, sampling :py:class:`torch.Tensor`'s for the underlying numerical data; by the duck-typing principle of Python/Pytorch dispatcher, conforming to the first requirement is usually a given if the second requirement is fullfilled).

We define a control module of appropriate call signature and return type (where for this exposition we stick to basic feedforward neural networks as provided by the :py:class:`~ml_adp.utils.fnn.FNN`-class included in :py:mod:`ml_adp`'s utilities)::

    >>> class FNNControl(torch.nn.Module):
    ...     def __init__(self, hidden_layer_size):
    ...         super().__init__()
    ...
    ...         size = (n_1 + n_2, hidden_layer_size, m)
    ...         self.fnn = FNN.from_config(size, hidden_activation='ReLU')
    ...
    ...     def forward(self, state_1, state_2):
    ...         state = torch.cat([state_1, state_2], dim=-1)
    ...         return {'control': self.ffn(state)}
    ... 

Choose a size for the single hidden layers and set the control functions::

    >>> hidden_layer_size = 20;
    >>> for time in range(len(cost_acc)):
    ...     cost_acc.control_functions[time] = FNNControl(hidden_layer_size)
    ...
    >>> cost_acc
    CostAccumulation(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       | FNNControl(       ... |      cost_func_0     
       1  |     state_func_1      | FNNControl(       ... |      cost_func_1     
       2  |     state_func_2      | FNNControl(       ... |      cost_func_2     
       3  |     state_func_3      | FNNControl(       ... |      cost_func_3     
       4  |     state_func_4      | FNNControl(       ... |      cost_func_4     
      (5) |         None          |                       |                      
    )


Numerical Simulation of Optimal Control Problems
------------------------------------------------

``cost_acc`` is now ready to perform the numerical simulation of the optimal control problem.
Assume, using the ongoing example, that 

* ``initial_state`` is a sample of the initial state $S_0$ of size $N$ (meaning that :pycd:`initial_state` is a dictionary whose :pycd:`"state_1"`- and :pycd:`"state_2"`-values are :py:class:`torch.Tensor`'s of size $(N, n_1)$ and $(N, n_2)$, respectively)

* ``random_effects`` is a list of samples of the random effects $\Xi_1, \dots, \Xi_T$ of sizes $N$ (meaning that all entries of ``random_effects`` are again dictionaries with appropriate keys and values that are :py:class:`torch.Tensor`'s of appropriate sizes)

Then

.. code::

    >>> cost = cost_acc(initial_state, random_effects)

produces the corresponding sample of the total cost $K^{F, A}(S_0, \Xi)$:
``cost`` is a tensor of size $(N, 1)$ and is aligned with the values of ``initial_state`` and of the entries of ``random_effects`` along the first axis (the *batch axis*). 
Now, if $N$ is a large-enough integer, then by the principle of Monte-Carlo simulation (the *law of large numbers*, more precisely) one can expect

.. code::

    >>> expected_cost = cost.mean()

to produce a numerical value close to $EK^{F, A}(S_0, \Xi)$.

Internally, :py:class:`~ml_adp.base.CostAccumulation` delegates the computation of the state evolution $S = (S_t)_{t = 0}^{T + 1}$ to the class :py:class:`~ml_adp.base.StateEvolution`, of which an instance it saves in its :py:attr:`~ml_adp.base.CostAccumulation.state_evolution`-attribute::

    >>> cost_acc.state_evolution
    StateEvolution(
     time |             state_func             |            control_func           
    ===============================================================================
       0  |                                    | FNNControl(                    ...
       1  |            state_func_1            | FNNControl(                    ...
       2  |            state_func_2            | FNNControl(                    ...
       3  |            state_func_3            | FNNControl(                    ...
       4  |            state_func_4            | FNNControl(                    ...
      (5) |                None                |                                   
    )

:py:class:`~ml_adp.base.StateEvolution`'s provide the state evolution (onto which, at each time, the respective controls are merged as well) through its :py:meth:`~ml_adp.base.StateEvolution.evolve`-method::

    >>> states = cost_acc.state_evolution.evolve(initial_state, random_effects)

In other words, ``states`` is now a list whose entries are the merged dictionaries of the respective states and controls $S_t$ and $A_t$ of the state evolution $S$ and the control $A$ (mathematically, the distinction between states and controls is purely semantic).

As a callable, :py:class:`~ml_adp.base.StateEvolution` implements the function $$F^A \colon (s_0, \xi) \mapsto s_{T+1} = F_T(\dots F_1(F_0(s_0, A_0(s_0), \xi_1), \dots) \dots , A_T(\dots), \xi_{T+1}),$$ returning only the post-problem state $s_{T + 1}$ of the state evolution and not the full state evolution.
This behavior permits to use :py:class:`~ml_adp.base.StateEvolution`'s as state functions.
In the current situation (with the post problem state function being :pycd:`None`),

.. code-block::

    >>> post_problem_state = cost_acc.state_evolution(initial_state, random_effects)

just sets ``post_problem_state`` to :pycd:`None` as well.


Naive Optimization
------------------

Optimizing all parameters of the control functions of ``cost_acc`` at once is viable approach at solving the optimal control problem.
The containers of the defining functions again all are Pytorch modules (they are :py:class:`~ml_adp.utils.nn.ModuleArray`'s, found in :py:mod:`ml_adp`'s :py:mod:`~ml_adp.utils.nn`-utilities module).
With such being so, passing all parameters of ``cost_acc`` to a Pytorch optimizer is as easy as::

    >>> optimizer = torch.optim.AdamW(cost_acc.control_functions.parameters())

Now, after execution of the following code, it is reasonable to expect ``cost_acc`` to have (more or less) optimal control functions.

.. literalinclude:: ./algos/naive.py

Here, ``gradient_descent_steps`` is a reasonable number $K$ of optimization iterations while ``initial_state_sampler`` and ``random_effects_sampler`` should produce samples of $S_0$ and $(\Xi_1, \dots, \Xi_T)$, respectively, in terms of the simulation size $N$.


The Dynamic Programming Principle
---------------------------------

The well-known dynamic programming principle allows to reduce the complexity of the numerical simulation in control function optimization by ensuring that tackling the problem step-wisely produces equivalent results.
Moreover, it promises explicit access to optimal controls to be of practical, scenario-wise applicability.

Defining (subsuming the right technical conditions)
$$V_t(s_t) = \inf_{a_t\in\mathbb{R}^{m_t}} Q_t(s_t, a_t)$$
$$Q_t(s_t, a_t) = K_t(s_t, a_t) + E(V_{t+1}(F_t(s_t, a_t, \Xi_{t+1}))) $$
backwards in time $t$ it is easily seen that $EV_0(S_0)$ constitutes a lower bound for the cost-to-go of the problem and that a control $\bar{A}$ must be optimal if, together with its state evolution $\bar{S}$, it satisfies
$$\bar{A}_t \in \operatorname{arg\,min}_{a_t\in\mathbb{R}^{m_t}} Q_t(\bar{S}_t, a_t).$$

This principle is known as the *dynamic programming principle* and the equations are called the *Bellman equations*.
They motivate a wide range of numerical algorithms that rely on computing the $V$- and $Q$-functions in a backwards-iterative manner as a first step and determining the optimal controls in a forward pass through the argmin-condition as a second step.

In the well-behaved situation (which in particular includes some regularity conditions for the state and cost functions), it can in turn be argued that, subtly, optimal controls $\bar{A}$ (if they exist) turn $EK^{F, \bar{A}}(S_0, \Xi)$ into $EV_0(S_0)$.
This result can be leveraged to formulate algorithms that, using machine learning methods, produce optimal controls as part of the backward pass, eliminating the need for a subsequent forward pass.
To see this, we first make explicit the simple fact that contiguous subcollections of the defining functions of an optimal control problem again give valid optimal control problems (of a shorter length) for which the above results obviously apply as well:
For all times $t$ we write $\xi_{t, T}$ for $(\xi_t, \dots, \xi_T)$ and $K^{F, A}_{t,T}(s_t, \xi_{t+1, T+1})$ for the total cost function belonging to the optimal control problem given by the state functions $(F_t, \dots, F_T)$, the cost functions $(K_t, \dots, K_T)$, and the control functions $(A_t, \dots, A_T)$.
Using this notation, it can be formulated that a suite of controls $\bar{A}$ is optimal if for all times $t$ $EK_{t, T}^{F, \bar{A}}(\bar{S}_t, \Xi_{t+1, T})$ is minimal when associated with $\bar{A}_t$ (meaning that for all controls $A = (A_0, \dots, A_T)$ for which $A_{t + 1} =\bar{A}_{t + 1}, \dots, A_T=\bar{A}_T$ have $EK_{t,T}^{F, A}(\bar{S}_t, \Xi_{t+1,T})\geq EK_{t,T}^{F, \bar{A}}(\bar{S}_t, \Xi_{t + 1,T})$), effectively turning $$K_{t - 1}(s_{t - 1}, a_{t - 1}) + E(K_{t, T}^{F, \bar{A}}(F_{t - 1}(s_{t - 1}, a_{t - 1}, \Xi_t), \Xi_{t + 1, T}))$$ into $Q_{t - 1}(s_{t - 1}, a_{t - 1})$ (which is key).
Some additional mathematical considerations allow to replace $\bar{S}_t$ (which at time $t$ of is not yet available if iterating backwards) in the above by some $\hat{S}_t$ sampled independently from $\Xi_{t + 1, T}$ and from a suitable *training distribution* and finally make an explicit backwards-iterative algorithm possible.

:py:class:`~ml_adp.base.CostAccumulation` implements the :pycd:`__getitem__`-method in a way that makes :pycd:`cost_acc[time:]` implement $K_{t, T}^{F, A}(s_0, \xi_{t + 1, T})$ (if ``time`` corresponds to $t$):

.. autofunction:: ml_adp.base.CostAccumulation.__getitem__

For example::

    >>> cost_acc[2:]
    CostAccumulation(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       | FNNControl(       ... |      cost_func_2     
       1  |     state_func_3      | FNNControl(       ... |      cost_func_3     
       2  |     state_func_4      | FNNControl(       ... |      cost_func_4     
      (3) |         None          |                       |                      
    )

Or::

    >>> cost_acc[-1]
    CostAccumulation(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       | FNNControl(       ... |      cost_func_4     
      (1) |         None          |                       |                      
    )


In terms of :py:mod:`ml_adp`, the algorithm as suggested above consists out of stepping backwards through time and optimizing, at each ``time``, the first control function of :pycd:`cost_acc[time:]` with the objective being :pycd:`cost_acc[time:](state, random_effects).mean()` for samples ``state`` and ``random_effects`` of the training state $\hat{S}_t$ and the random effects $\Xi_{t+1, T}$, respectively:

.. literalinclude:: ./algos/nn_contpi.py
    :lines: 5-53
    :linenos:

Here, ``training_state_samplers`` and ``random_effects_samplers`` are lists containing the samplers of the relevant random variables (indicated in the code comments).
``Optimizer``, instead, is short for some :py:class:`torch.optim.Optimizer`.

To see the algorithm - which `here`_ has been introduced as the *NNContPI* algorithm - in action, look at the `option hedging example`_.

.. _here: https://arxiv.org/abs/1812.05916v3

.. _option hedging example: examples/option_hedging.html#NNContPI

Value Function Approximation and Hybrid Methods
-----------------------------------------------

The above algorithm has the shortcoming of the numerical simulation getting more and more complex as the backward pass advances.
The technique of *value function approximation* alleviates this issue and, in the present context, consists of replacing, at each ``time`` of the NNContPI algorithm, the tail part of ``objective`` by an (approximately) equivalent other :py:class:`~ml_adp.base.CostAccumulation` that is computationally more efficient.


.. Entering step ``step``, the ``cost_acc[step+1]`` assumedly implements $V_{t+1}(s_{t+1}, \xi_{t+1})$ (on the support of the training distribution, that is) such that if some other :py:class:`ml_adp.base.CostAccumulation` implements an approximation $\tilde{V}_t(s_t, \xi_t)$ of$EV_t(s_t, \xi_t, \Xi_{t+1, T})$

Mathematically, if for time $t$ some $\tilde{V}_{t+1}(s_{t+1})$ is an approximation of $V_{t+1}(s_{t+1})$, then optimizing $A_t$ in 
$$E(K_t(\hat{S}_t, A_t) + \tilde{V}_{t+1}(F_t(\hat{S}_t, A_t, \Xi_{t+1})))$$
can still be expected to lead to an (approximately) optimal control $\bar{A}_t$ for the original problem.

The composition-decomposition properties of :py:class:`~ml_adp.base.CostAccumulation` make it easy to perform a such replacement of the time-$(t+1)$ $V$-function ($V$ functions are called *value functions* in the literature where a *value-based* formulation of optimization problems is more common).
In addition to :py:class:`~ml_adp.base.CostAccumulation` implementing the :pycd:`__getitem__`-method as explained in the previous section, it implements the :pycd:`__add__`-method:

.. autofunction:: ml_adp.base.CostAccumulation.__add__

It does so in a way that makes :py:meth:`~ml_adp.base.CostAccumulation.__getitem__` and :py:meth:`~ml_adp.base.CostAccumulation.__add__` work together consistently.
For example, for any time ``time``::

    >>> cost_acc[:time] + cost_acc[time:]  # Split and re-compose
    CostAccumulation(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       | FNNControl(       ... |      cost_func_0     
       1  |     state_func_1      | FNNControl(       ... |      cost_func_1     
       2  |     state_func_2      | FNNControl(       ... |      cost_func_2     
       3  |     state_func_3      | FNNControl(       ... |      cost_func_3     
       4  |     state_func_4      | FNNControl(       ... |      cost_func_4     
      (5) |         None          |                       |                      
    )

So, in terms of :py:mod:`ml_adp`, it can be formulated that value function approximation consists of replacing, at ``time``, ``cost_acc[time:]`` by ``cost_acc[time] + cost_approximator`` where ``cost_approximator`` is another :py:class:`~ml_adp.base.CostAccumulation` that approximates ``cost_acc[time+1:]``.

``cost_approximator`` could be a zero-step, non-controlled :py:mod:`~ml_adp.base.CostAccumulation`::

    >>> cost_approximator
    CostAccumulation(
     time |             state_func             |             cost_func             
    ===============================================================================
       0  |                                    |             FNNCost()             
      (1) |                None                |                                   
    )

Here, it is implied that the time-0 cost function of ``cost_approximator`` is neural network based and has been trained beforehand to accurately implement the relevant value function.
Notice also that the representation of ``cost_approximator`` does not feature a ``control_func``-column, indicating the internal management of control functions to be skipped entirely.
This behavior can be set using the :py:attr:`~ml_adp.base.CostAccumulation.controlled`-attribute::
    
    >>> cost_approximator.controlled
    False`

Continuing::

    >>> cost_acc_approximator = cost_acc[0] + cost_approximator
    >>> cost_acc_approximator
    CostAccumulation(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       | FNNControl(       ... |      cost_func_0     
       1  |     state_func_1      |         None          |       FNNCost()      
      (2) |         None          |                       |                      
    )

Now, ``cost_acc_approximator`` is numerically equivalent to ``cost_acc`` (presuming that the neural network based cost function of ``cost_approximator`` in fact approximates $(s_1, \xi_{2, T}) \mapsto EK_{1,T}^{F, A}(s_1, \xi_{2,T})$ well) while the single-step ``cost_acc_approximator`` may well be much more computationally efficient than the multi-step ``cost_acc``.

To additionally align the behaviors of the underlying :py:class:`~ml_adp.base.StateEvolution`'s of' ``cost_acc`` and ``cost_acc_approximator`` one could set the post problem state function of ``cost_acc_approximator`` to the "missing" portion of ``cost_acc``'s :py:class:`~ml_adp.base.StateEvolution`::

    >>> cost_acc_approximator.state_functions[1] = cost_acc[1:].propagator
    >>> cost_acc_approximator
    CostAccumulation(
     time |      state_func       |     control_func      |       cost_func      
    =============================================================================
       0  |                       | FNNControl(       ... |      cost_func_0     
       1  |     state_func_1      |         None          |       FNNCost()      
      (2) | StateEvolution(   ... |                       |                      
    )

It is, however, usually advantageous to omit this step and not incure the computational cost of simulating the post-problem state.

The following algorithm introduces value function approximation to the NNContPI algorithm as explained above and `by the same authors`_ has been termed the *HybridNow* algorithm. 

.. _by the same authors: https://arxiv.org/abs/1812.05916v3

.. literalinclude:: ./algos/hybridnow.py
    :lines: 5-53
    :linenos:

To see this algorithm in action, look, again, at the `option hedging example notebook`_.

.. _option hedging example notebook: examples/option_hedging.html#HybridNow

Choosing the Training Distributions (WIP)
-----------------------------------------



.. 
    
    Approximate dynamic programming algorithms require the sampling of states from the so-called training distributions at each time.

The choice of the training distributions is an empiric process informed by the knowledge of a domain expert.

