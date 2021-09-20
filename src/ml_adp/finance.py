"""
Provides finance-related functionality.
"""


from abc import abstractmethod
from collections.abc import Callable
import torch
from torch.distributions import categorical


class MarketStep:
    r"""
    Provides callable instances that as state functions implement a portfolio single financial market step.

    Implements
    $$F((w, p), a, r)\colon (\mathbb{R}\times\mathbb{R}^n)\times\mathbb{R}^n \to \mathbb{R}^n\times \mathbb{R}
    ((w,p), r) \mapsto \left(w + (w-\sum_i a_i)r_0 + r'a, (1 + r)\cdot p\right)
    $$
    and as such asks that 
    
    * $(w, p)$ be the wealth of the portfolio agent and market prices of the $n$ risky assets at initial time
    * $a$ be the list of $n$ net absolute investments into the risky assets
    * $r$ be the list of excess return rate of the risky assets over the time step
    * $r_0$ to be the excess return rate of the riskless asset over the time step (as saved by :attr:`risk_free_rate`).

    Returns
    -------
    [type]
        [description]
    """
    
    risk_free_rate: float
    
    def __init__(self, risk_free_rate: float = 0.0) -> None:
        """
        Create a market step state function.

        Parameters
        ----------
        risk_free_rate : float, optional
            The rate of return provided by the risk-free asset at that step, by default 0.0
        """
        self.risk_free_rate = risk_free_rate
    
    def __call__(self, wp, position, excess_returns):
        wealth = wp[:, 0]
        market_price = wp[:, 1:]

        net_amount = torch.einsum('bj->b', position)
        bank_account = wealth - net_amount

        change_w = (bank_account * self.risk_free_rate
                     + torch.einsum('bj,bj->b', position, excess_returns))
        change_p = market_price * excess_returns

        new_wp = wp.clone()
        new_wp[:, 0] = wealth + change_w
        new_wp[:, 1:] = market_price + change_p

        return new_wp


class Derivative(Callable):
    
    @abstractmethod
    def __call__(self, wps, positions):
        return
    
    
class SquareReplicationError:
    
    derivative: Derivative

    def __init__(self, derivative: Derivative) -> None:
        self.derivative = derivative
    
    def __call__(self, wp: torch.Tensor, position: torch.Tensor, excess_returns=None) -> torch.Tensor:
        # wps=state, positions=control, no random_effects
        wealth = wp[:, [0]]
        payoff = self.derivative(wp, position)
        return (payoff - wealth).square()

    
class EuropeanCallOption(Derivative):
    
    strike: float
    underlying_index: int
    
    def __init__(self, strike: float, underlying_index: int) -> None:
        super().__init__()
        
        self.strike = strike
        self.underlying_index = underlying_index
    
    def __call__(self, wp, position):
        underlying_price = wp[:, [self.underlying_index]]
        option_payoff = torch.maximum(
            underlying_price - self.strike,
            torch.tensor(0.0)
        )
        return option_payoff


class MultinomialReturnsSampler:
    def __init__(self,
                 returns,
                 probabilities,
                 number_iid_assets=1):

        self.categorical = categorical.Categorical(
            probs=probabilities
        )
        self.returns = returns
        self.number_iid_assets = number_iid_assets
        # TODO Support non-identically distributed assets

    def __call__(self, simulations, length):
        rets_sample = [
            self.returns[
                self.categorical.sample([simulations, self.number_iid_assets])
            ]
            for step in range(length)
        ]

        return rets_sample
    
        