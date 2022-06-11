r"""Machine learning based ADP for finite-horizon discrete-time stochastic optimal control problems

    An optimal control problem within the scope of :mod:`ml_adp` of $T$ *steps* consists of $T+1$ *state functions*
    $$F_0(s_0,a_0,\xi_1),\dots ,F_T(s_T, a_T, \xi_{T+1}),$$
    together with $T+1$ *cost functions*
    $$K_0(s_0, a_0, \xi_0),\dots, K_T(s_T, a_T).$$

    Given an initial value $S_0$ and independent "random effects" $\Xi_1,\dots, \Xi_{T+1}$, a "dynamic" choice of controls $A_0,\dots, A_T$ defines a controlled random *state evolution*
    $$S_{t+1} = F_{t+1}(S_t, A_t, \Xi_{t+1}), \quad t=0,\dots, T$$
    with which the random *total cost*
    $$K^{F, A}(S_0, \Xi_1, \dots, \Xi_{T+1}) = \sum_{t=0}^T K_t(S_t, A_t)$$
    is associated.
    A choice of controls is optimal, if, in prospective expectation, the associated cost is minimal.

    The theory establishes in sufficiently well-behaved situations that choices of controls
    in the form of *control functions* $A_0(s_0), \dots, A_T(s_T)$ (via $A_t = A_t(S_t)$)
    from function classes within which common neural network architectures have universal approximation capabilities exhibit the required dynamicity and are not restricted in regard to their potential optimality.

    :mod:`ml_adp` wraps the state functions $F=(F_0,\dots, F_T)$ and a choice of controls $A=(A_0,\dots, A_T)$ into :class:`ml_adp.cost.Propagator`'s to provide the numerical simulation of the state evolution and bundles a such :class:`ml_adp.cost.Propagator`-instance with the data $K=(K_0,\dots, K_T)$ into :class:`ml_adp.cost.CostToGo`'s to provide the numerical simulation of the total cost.
"""

__version__ = "0.3.0a0"