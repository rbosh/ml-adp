r"""
:mod:`ml_adp` is a Pytorch-enabled Python library for the ADP-based solution of finite-horizon discrete optimal control problems

An optimal control problem as in the scope of :mod:`ml_adp` consists of $T$ random state functions
$$F_1(s_0,a_0,\xi_1),\dots ,F_{T+1}(s_T, a_T, \xi_{T+1}),$$
of $T$ control functions
$$A_0(s_0),\dots, A_T(s_T),$$
and of $T$ random cost functions
$$h_0(s_0, a_0, \xi_0),\dots, h_T(s_T, a_T, \xi_T),$$
all of which the user is tasked to furnish to the library as Python `callable`'s.

The user may then provide samples of the initial state $s_0$ and the random effects $\xi_0,\dots, \xi_{T+1}$
and be returned the state evolution $(s_i)_{i=0}^{T+1}$ and the computed control $(a_i)_{i=0}^T$ as specified per
$s_{i+1} = S_{i+1}(s_i, a_i, \xi_{i+1}),\quad a_i = A_i(s_i), \quad i=0,\dots, T$
and computed by :class:`Propagator` as well as the sample of the cost
$$\sum_{i=0}^T h_i(s_i, a_i, \xi_i)$$
as computed by :class:`CostToGo`.
"""

__version__ = "0.2.2"