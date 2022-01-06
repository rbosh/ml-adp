"""
Provides Parametrized Linear and Related Mappings
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple, Dict, Union

import torch
import torch.nn as nn
from ml_adp.nn import FFN, Layer, MultiHead  # TODO Make this work: , FFNDims


SpaceSize = Union[int, Sequence[int]]
FFNDims = Sequence[SpaceSize]
                   
# TODO Add test that checks flatten and view to be inverses

def _batch_dot(inputs1, inputs2):
    return torch.einsum(
        'bj,bj->b',
        inputs1,
        inputs2
    ).unsqueeze(1)


class ConstantMap(nn.Module):
    r"""
    Parametrizable Input-Constant Mapping

    Given a parametrized value $(c_p)$, implements
    $$(x, p)\mapsto c_p.$$
    """
    def __init__(self, value_rep: Callable, default_param: Optional[torch.Tensor] = None):
        r"""
        Construct Constant Mapping

        Parameters
        ----------
        value_rep : Callable
            The parametrized value $(c_p)$
        default_param:
            Default parameter value $p_0$ (implies $(c_p)_p$ to be constant in $p$) to pass, by default None
        """
        super().__init__()

        self.value_rep = value_rep
        #self._default_param = torch.empty(1, 0) if isinstance(value_rep, FFN) else None
        self.default_param = default_param
    
    def forward(self, 
                input_: Optional[torch.Tensor] = None,
                param: Optional[torch.Tensor] = None):
        param = param if param is not None else self.default_param
        return self.value_rep(param)

    @classmethod
    def from_tensorrep(cls, tensorrep: torch.Tensor) -> ConstantMap:
        r"""
        Construct constant mapping, constant in parameter as well, from unique value

        Returns :class:`ConstantMap` implementing
        $$(x, p)\mapsto c_0$$
        where $c_0$ is given by `tensorrep`

        Parameters
        ----------
        tensorrep : torch.Tensor
            The constant value $c_0$

        Returns
        -------
        ConstantMap
            The constant map $(x, p)\mapsto c_0$
        """
        rep = FFN.from_config(sizes=(0, tensorrep.size()))
        rep[0].linear.bias.data = tensorrep.flatten()
        for param in rep.parameters():
            param.requires_grad = False
            
        return ConstantMap(value_rep=rep, default_param=torch.empty(1, 0))
    

class LinearMap(nn.Module):
    r"""
    Implements a parametrized linear map with translation and bias.

    Implements for all parameters $p$
    $$x\mapsto A_p(x-s_p) + b_p$$
    where $A_p$ is linear map parametrized by $p$,
    $s_p$ is a translation vector parametrized by $p$,
    and $b_p$ is a bias vector parametrized by $p$.
    """
    def __init__(self,
                 linear_rep: Callable[[Optional[torch.Tensor]], Tuple[Optional[torch.Tensor]]],
                 default_param: Optional[torch.Tensor] = None) -> None:
        r"""
        Construct a Linear Map

        Parameters
        ----------
        linear_rep : Callable[[Optional[torch.Tensor]], Tuple[Optional[torch.Tensor]]]
            Should give $p\mapsto (A_p, b_p, s_p)$
        default_param : Optional[torch.Tensor], optional
            Default parameter value, by default None
        """
        super().__init__()

        self.linear_rep = linear_rep
        self.default_param = default_param

    def forward(self,
                input_: torch.Tensor,
                param: Optional[torch.Tensor] = None) -> torch.Tensor:

        param = param if param is not None else self.default_param
        param = self.linear_rep(param)

        if param[2] is not None:  # Translation
            input_ = input_ - param[2]

        if param[0] is not None:  # Linear Operation
            input_ = torch.einsum(
                'b...j,bj->b...',
                param[0],
                input_
            )

        if param[1] is not None:  # Bias
            input_ = input_ + param[1]

        return input_

    @classmethod
    def from_ffnconfig(self,
                       dims: FFNDims,
                       bias: bool = True,
                       translate: bool = False,
                       ffn_config: Optional[Dict] = None) -> LinearMap:

        if ffn_config is None:
            ffn_config = {}

        dims = list(dims)
        matrix_shape = dims.pop(-1)  # SIDE EFFECT
        in_features = matrix_shape[0]
        out_features = matrix_shape[0]

        ffn = FFN.from_config(dims, **ffn_config)
        linear_rep = Layer(dims[-1], matrix_shape, **ffn_config)
        bias_rep = Layer(dims[-1], out_features, **ffn_config) if bias else None
        translation_rep = Layer(dims[-1], in_features, **ffn_config) if translate else None

        return LinearMap(nn.Sequential(ffn, MultiHead(linear_rep, bias_rep, translation_rep)))

    @classmethod
    def from_tensorrep(cls,
                       linear_tensorrep: Optional[torch.Tensor] = None,
                       bias_tensorrep: Optional[torch.Tensor] = None,
                       translate_tensorrep: Optional[torch.Tensor] = None):
        # TODO Add parameter freeze=True

        if linear_tensorrep is not None:
            linear_rep = Layer(0, linear_tensorrep.size())
            linear_rep.linear.bias.data = linear_tensorrep.flatten()
            for param in linear_rep.parameters():
                param.requires_grad = False
        else:
            linear_rep = None

        if bias_tensorrep is not None:
            bias_rep = Layer(0, bias_tensorrep.size(0))
            bias_rep.linear.bias.data = bias_tensorrep.flatten()
            for param in bias_rep.parameters():
                param.requires_grad = False
        else:
            bias_rep = None

        if translate_tensorrep is not None:
            translate_rep = Layer(0, translate_tensorrep.size(0))
            translate_rep.linear.bias.data = translate_tensorrep.flatten()
            for param in translate_rep.parameters():
                param.requires_grad = False
        else:
            translate_rep = None

        return LinearMap(MultiHead(linear_rep, translate_rep, bias_rep), default_param=torch.empty(1, 0))


class BilinearForm(nn.Module):
    r"""
    Parametrized Bilinear Form

    Saves parametrized linear maps :class:`LinearMap` $B_1=(B_{1, p})_p$ and $B_2=(B_{2, p})_p$,
    and, as a callable, implements
    $$(x_1, x_2, p)\mapsto x_2^*(B^*_{2, p}B_{1, p}x_1).$$
    """

    def __init__(self, linear_rep1: LinearMap, linear_rep2: LinearMap):
        r"""
        Construct Bilinear Form

        Parameters
        ----------
        linear_rep1 : LinearMap
            Corresponds to $B_1$
        linear_rep2 : LinearMap
            Corresponds to $B_2$
        """
        super(BilinearForm, self).__init__()
        self.linear1 = linear_rep1
        self.linear2 = linear_rep2

    def forward(self, input_tuple, params):
        intermediates1 = self.linear1(input_tuple[0], params)
        intermediates2 = self.linear2(input_tuple[1], params)
        return _batch_dot(intermediates1, intermediates2)


class Semi2Norm(nn.Module):

    r"""Typically a squared (semi) norm.
    
    As a callable, implements
    $$x\mapsto x'B'Bx$$
    or, in other words, the quadratic form $Q$ with 
    matrix representant $Q=B'B$ which is a semi norm and a norm
    whenever $Q'Q$ is positive definite.
    """

    def __init__(self, input_space_trafo_rep: LinearMap) -> None:
        r"""
        Construct a squared semi norm object

        Parameters
        ----------
        input_space_trafo_rep : LinearMap
            Corresponds to :math:`B`
        """
        super(Semi2Norm, self).__init__()
        self.bilinear_rep = BilinearForm(input_space_trafo_rep,
                                         input_space_trafo_rep)

    def forward(self, input_, param: Optional[torch.Tensor] = None):
        return self.bilinear_rep((input_, input_), param)
    
    @classmethod
    def from_sym_tensorrep(cls, sym_rep: torch.Tensor) -> Semi2Norm:
        """
        Gives a :class:`Semi2Norm` with matrix representant $Q$ given by :math:`sym_rep`

        [extended_summary]

        Parameters
        ----------
        sym_rep : torch.Tensor
            Corresponds to :math:`Q`

        Returns
        -------
        Semi2Norm
            The squared norm
        """
        # Don't forget the batch dimension
        eig, base_trafo = torch.linalg.eigh(sym_rep, UPLO="U")
        diag = torch.diag(eig.sqrt())

        input_space_trafo_rep = torch.einsum(
            'ij,jk->ki',  # matrix mult. and transpose
            base_trafo,
            diag
        )
        return Semi2Norm(LinearMap.from_tensorrep(input_space_trafo_rep))


class QuadraticStepCost(nn.Module):

    def __init__(self,
                 state_space_norm: Semi2Norm,
                 control_space_norm: Semi2Norm,
                 step_size: float):
        super(QuadraticStepCost, self).__init__()
        self.state_cost = state_space_norm
        self.control_cost = control_space_norm
        self.step_size = step_size

    @classmethod
    def from_sym_tensorrep(cls,
                           state_norm_rep,
                           control_norm_rep,
                           step_size):

        state_space_norm = Semi2Norm.from_sym_tensorrep(state_norm_rep)
        control_space_norm = Semi2Norm.from_sym_tensorrep(control_norm_rep)

        return QuadraticStepCost(state_space_norm,
                                 control_space_norm,
                                 step_size)

    def forward(self, state, control, parameter_state=None):
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