"""
Provides Linear-Quadratic Optimal Control Problem Components
"""
import math

import torch
from ml_adp.mapping.linear import QuadraticMap, LinearMap
from torch import nn as nn


class QuadraticCost(nn.Module):

    def __init__(self,
                 state_space_norm: QuadraticMap,
                 control_space_norm: QuadraticMap,
                 step_size: float):
        super(QuadraticCost, self).__init__()
        self.state_cost = state_space_norm
        self.control_cost = control_space_norm
        self.step_size = step_size

    @classmethod
    def from_sym_tensorrep(cls,
                           state_norm_rep,
                           control_norm_rep,
                           step_size):

        state_space_norm = QuadraticMap.from_sym_tensorrep(state_norm_rep)
        control_space_norm = QuadraticMap.from_sym_tensorrep(control_norm_rep)

        return QuadraticCost(state_space_norm,
                             control_space_norm,
                             step_size)

    def forward(self, state, control):
        parameter_state = torch.empty(1, 0)
        states_cost = self.state_cost(state, parameter_state)
        controls_cost = self.control_cost(control, parameter_state)
        return self.step_size * (states_cost + controls_cost)


class LinearControlledStep(nn.Module):

    def __init__(self,
                 linear_rep_state: torch.Tensor,
                 linear_rep_control: torch.Tensor,
                 linear_rep_rand: torch.Tensor,
                 step_size: float) -> None:
        super(LinearControlledStep, self).__init__()

        self.B = LinearMap.from_tensorrep(linear_rep_state)
        self.C = LinearMap.from_tensorrep(linear_rep_control)
        if linear_rep_rand.dim() != 3:
            raise ValueError("`linear_rep_rand` must map to linear maps, i.e. must have dim=3")
        self.D = LinearMap.from_tensorrep(linear_rep_rand)
        self.step_size = step_size

    def forward(self, state, control, random_effect):

        # TODO Extract this to init
        parameter_state = torch.empty(1, 0)

        drift_from_state = self.B(state, parameter_state)
        drift_from_control = self.C(control, parameter_state)
        controlled_diffusion_term = self.D(control, parameter_state)

        random_increment = torch.einsum(
            'b...j,bj->b...',
            controlled_diffusion_term,
            random_effect
        )

        return (state
                + self.step_size * (drift_from_state + drift_from_control)
                + math.sqrt(self.step_size) * random_increment)