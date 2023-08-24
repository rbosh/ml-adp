r"""Machine learning based ADP for finite-horizon discrete-time stochastic optimal control problems

    An optimal control problem within the scope of :mod:`ml_adp` of $T$ *steps* consists of $T+1$ *state functions*
    $$\mathrm{state\_func}_0(s_0, a_0, \xi_1), \dots, \mathrm{state\_func}_T(s_T, a_T, \xi_{T+1})$$
    together with $T+1$ *cost functions*
    $$\mathrm{cost\_func}_0(s_0, a_0), \dots, \mathrm{cost\_func}_T(s_T, a_T)$$

    Given an initial value $S_0$ and independent "random effects" $\Xi_1,\dots, \Xi_{T+1}$, a "dynamic" choice of controls $A_0,\dots, A_T$ defines a random *controlled state evolution*
    $$S_{t+1} = \mathrm{state\_func}_t(S_t, A_t, \Xi_{t+1}),\quad t=0,\dots, T$$
    with which the random *total cost*
    $$\mathrm{cost\_accumulation}(S_0, \Xi_1, \dots, \Xi_{T+1}) = \sum_{t=0}^T \mathrm{cost\_func}_t(S_t, A_t)$$
    is associated.
    A choice of controls is optimal, if, in prospective expectation, the associated cost is minimal.

    The theory establishes in sufficiently well-behaved situations that choices of controls
    in *feedback form*, i.e., in the form of *control functions* 
    $$\mathrm{control\_func}_0(s_0), \dots, \mathrm{control\_func}_T(s_T)$$
    (via $A_t = \mathrm{control\_func}_t(S_t)$) from function classes within which common neural network architectures have universal approximation capabilities exhibit the required dynamicity and are potentially optimal.

    :mod:`ml_adp` wraps the state functions and a choice of control functions into :class:`ml_adp.base.StateEvolution`'s to provide the numerical simulation of the state evolution and bundles these :class:`ml_adp.base.StateEvolution`-instances together with the cost functions into :class:`CostAccumulations`'s to provide the numerical simulation of the total cost incurred along the underlying state evolution.
    Both :class:`ml_adp.base.StateEvolution` and :class:`ml_adp.base.CostAccumulation` subclass Pytorch's :class:`torch.nn.Module`-class and appropriately register the parameters of neural network based control functions as their own, opening up the simulation of the total cost to Pytorch's automatic differentiation capabilities.
"""

__version__ = "0.3.0a1"

from .base import StateEvolution, CostAccumulation
