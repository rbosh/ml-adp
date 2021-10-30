r"""
:mod:`ml_adp` is a Pytorch-enabled Python library for the ADP-based solution of finite-horizon discrete-time stochastic optimal control problems

An optimal control problem within the scope of :mod:`ml_adp` consists of $T$ random state functions
$$F_1(s_0,a_0,\xi_1),\dots ,F_{T+1}(s_T, a_T, \xi_{T+1}),$$
together $T$ random cost functions
$$h_0(s_0, a_0, \xi_0),\dots, h_T(s_T, a_T, \xi_T).$$

Given an initial value $s_0$ and "random effects" $\xi_0,\dots, \xi_{T+1}$, a "dynamic" choice of 
controls $a_0,\dots, a_T$ defines a controlled state evolution
$$s_{t+1} = F_{t+1}(s_t, a_t, \xi_{t+1}), \quad t=0,\dots, T$$
with which the cost
$$\sum_{t=0}^T h_t(s_t, a_t, \xi_t)$$
is associated.

The theory establishes that control choices
$$a_t = A_t(s_t, \xi_t), \quad t=0,\dots, T$$
for "control functions" $A_0(s_0, \xi_0), \dots, A_T(s_T, \xi_T)$ fulfill the required dynamicity of the choice and that,
in sufficiently regular situations, there are such functions that produce optimal choices for all initial values $s_0$ and random
effects $\xi_0,\dots, \xi_{T+1}$, and that such functions can be found within function classes within which common neural network architectures
exhibit universal approximation capabilities.
Importantly, control functions are optimal, if they are so on average, allowing Monte-Carlo gradient-descent methods.

The user may then provide samples of the initial state $s_0$ and the random effects $\xi_0,\dots, \xi_{T+1}$
and be returned the state evolution $(s_i)_{i=0}^{T+1}$ and the computed control $(a_i)_{i=0}^T$ as specified per
$s_{i+1} = S_{i+1}(s_i, a_i, \xi_{i+1}),\quad a_i = A_i(s_i), \quad i=0,\dots, T$
and computed by :class:`Propagator` as well as the sample of the cost
$$\sum_{i=0}^T h_i(s_i, a_i, \xi_i)$$
as computed by :class:`CostToGo`.
"""

__version__ = "0.2.2"