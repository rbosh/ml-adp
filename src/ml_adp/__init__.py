r"""
:mod:`ml_adp` is a Pytorch-enabled Python micro-library for the ADP-based solution of finite-horizon discrete-time stochastic optimal control problems

An optimal control problem within the scope of :mod:`ml_adp` consists of $T$ *state functions*
$$F_1(s_0,a_0,\xi_1),\dots ,F_{T+1}(s_T, a_T, \xi_{T+1}),$$
together with $T$ *cost functions*
$$h_0(s_0, a_0, \xi_0),\dots, h_T(s_T, a_T, \xi_T).$$

Given an initial value $S_0$ and independent "random effects" $\Xi_0,\dots, \Xi_{T+1}$, a "dynamic" choice of 
controls $A_0,\dots, A_T$ defines a controlled random state evolution
$$S_{t+1} = F_{t+1}(S_t, A_t, \Xi_{t+1}), \quad t=0,\dots, T$$
with which the random cost
$$h^{F, A}(S_0, \Xi_0, \dots, \Xi_{T+1}) = \sum_{t=0}^T h_t(S_t, A_t, \Xi_t)$$
is associated.
A choice of controls is optimal, if, in prospective expectation, the associated cost is minimal.

The theory establishes in sufficiently well-behaved situations that choices of controls
in the form of *control functions* $A_0(s_0, \xi_0), \dots, A_T(s_T, \xi_T)$ (via $A_t = A_t(S_t, \Xi_t)$)
from function classes within which common neural network architectures have universal approximation capabilities
exhibit the required dynamicity and are not restricted in regards to their potential optimality.

:mod:`ml_adp` organizes a given set of state, control, and cost functions as 
instances of its defined class :class:`CostToGo`.
"""

__version__ = "0.2.2"