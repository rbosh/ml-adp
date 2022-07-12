"""
Provides Parametrized Linear and Related Mappings
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from ml_adp.nn import FFN, Layer, MultiHead  # TODO Make this work: , FFNDims


SpaceSize = Union[int, Sequence[int]]
FFNSize = Sequence[SpaceSize]
                   

def _batch_dot(inputs1, inputs2):
    return torch.einsum(
        'bj,bj->b',
        inputs1,
        inputs2
    ).unsqueeze(1)


class ConstantMap(nn.Module):
    r"""
    Parametrizable Input-Constant Mapping

    Given a parametrized value $(c_\eta)$, implements
    $$(x, \eta)\mapsto c_{\eta}.$$
    """
    def __init__(self, value_rep: Callable, default_param: Optional[torch.Tensor] = None):
        r"""
        Construct Constant Mapping

        Parameters
        ----------
        value_rep : Callable
            The parametrized value $(c_{\eta})$
        default_param:
            Default parameter value $\eta_0$ (implies $(\eta_p)_p$ to be constant in $\eta$) to pass, by default None
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
        $$(x, \eta)\mapsto c_0$$
        where $c_0$ is given by `tensorrep`

        Parameters
        ----------
        tensorrep : torch.Tensor
            The constant value $c_0$

        Returns
        -------
        ConstantMap
            The constant map $(x, \eta)\mapsto c_0$
        """
        rep = FFN.from_config(size=(0, tensorrep.size()))
        rep[0].linear.bias.data = tensorrep.flatten()
        for param in rep.parameters():
            param.requires_grad = False
            
        return ConstantMap(value_rep=rep, default_param=torch.empty(1, 0))
    

class LinearMap(nn.Module):
    r"""
    Implements Input-Linear Map With Translation and Bias.

    Saves a family of linear map representatives $(A_{\eta}, b_{\eta}, v_{\eta})_{\eta}$ in the sense that
        
    * $A_{\eta}\in\mathbb{R}^{m\times k}$ (``None`` indicates the $A_{\eta}$ being the identity
    * $b_{\eta}\in\mathbb{R}^m$ (``None`` indicates $b_{\eta}=0$)
    * $v_{\eta}\in\mathbb{R}^k$ (``None`` indicates $v_{\eta}=0$)
    
    As a callable, implements
    $$\mathbb{R}^k\to\mathbb{R}^m,\quad (x, \eta)\mapsto A_{\eta}(x - v_{\eta}) + b_{\eta}.$$
    """
    def __init__(self,
                 linear_rep: Callable[[Optional[torch.Tensor]], Tuple[Optional[torch.Tensor]]],
                 default_param: Optional[torch.Tensor] = None) -> None:
        r"""
        Construct a Linear Map

        Parameters
        ----------
        linear_rep : Callable[[Optional[torch.Tensor]], Tuple[Optional[torch.Tensor]]]
            The representative $\eta\mapsto (A_{\eta}, b_{\eta}, v_{\eta})$
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
                       size: FFNSize,
                       bias: bool = True,
                       translate: bool = False,
                       **ffn_config) -> LinearMap:

        size = list(size)
        matrix_shape = size.pop(-1)  # SIDE EFFECT
        in_features = matrix_shape[1]
        out_features = matrix_shape[0]

        hidden_activation = ffn_config.get('hidden_activation', torch.nn.ELU())  # ELU is default FFN activation
        base_config = ffn_config.copy()
        base_config['output_activation'] = hidden_activation
    
        base_ffn = FFN.from_config(size, **base_config)
    
        linear_rep = Layer.from_config(size[-1], matrix_shape, **ffn_config)
        bias_rep = Layer.from_config(size[-1], out_features, **ffn_config) if bias else None
        translation_rep = Layer.from_config(size[-1], in_features, **ffn_config) if translate else None

        return LinearMap(nn.Sequential(base_ffn, MultiHead(linear_rep, bias_rep, translation_rep)))

    @classmethod
    def from_tensorrep(cls,
                       linear_tensorrep: Optional[torch.Tensor] = None,
                       bias_tensorrep: Optional[torch.Tensor] = None,
                       translate_tensorrep: Optional[torch.Tensor] = None):
        # TODO Add parameter freeze=True

        if linear_tensorrep is not None:
            linear_rep = Layer.from_config(0, linear_tensorrep.size(), batch_normalize=False)
            linear_rep.linear.bias.data = linear_tensorrep.flatten()
            for param in linear_rep.parameters():
                param.requires_grad = False
        else:
            linear_rep = None

        if bias_tensorrep is not None:
            bias_rep = Layer.from_config(0, bias_tensorrep.size(0), batch_normalize=False)
            bias_rep.linear.bias.data = bias_tensorrep.flatten()
            for param in bias_rep.parameters():
                param.requires_grad = False
        else:
            bias_rep = None

        if translate_tensorrep is not None:
            translate_rep = Layer.from_config(0, translate_tensorrep.size(0), batch_normalize=False)
            translate_rep.linear.bias.data = translate_tensorrep.flatten()
            for param in translate_rep.parameters():
                param.requires_grad = False
        else:
            translate_rep = None

        return LinearMap(MultiHead(linear_rep, translate_rep, bias_rep), default_param=torch.empty(1, 0))


class BilinearMap(nn.Module):
    r"""Parametrized Bilinear Map

    Saves :class:`LinearMap`'s $A^{(1)}=(A^{(1)}_{\eta})_{\eta}$ 
    and $A^{(2)}=(A^{(2)}_{\eta})_{\eta}$, and, as a callable, implements
    the bilinear form
    $$((x^{(1)}, x^{(2)}), \eta)\mapsto \left(A^{(2)}_{\eta} x^{(2)}\right)^{\top} \left(A^{(1)}_{\eta} x^{(1)}\right).$$
    """

    def __init__(self, linear_rep1: LinearMap, linear_rep2: LinearMap):
        r"""
        Construct Bilinear Form

        Parameters
        ----------
        linear_rep1 : LinearMap
            Implements $\eta\mapsto A^{(1)}_{\eta}$
        linear_rep2 : LinearMap
            Implements $\eta\mapsto A^{(2)}_{\eta}$
        """
        super(BilinearMap, self).__init__()
        self.linear1 = linear_rep1
        self.linear2 = linear_rep2

    def forward(self, input_tuple, params):
        intermediates1 = self.linear1(input_tuple[0], params)
        intermediates2 = self.linear2(input_tuple[1], params)
        return _batch_dot(intermediates1, intermediates2)


class QuadraticMap(nn.Module):

    r"""Parametrized Quadratic Map
    
    Essentially, saves a :class:`LinearMap` $(A_{\eta})_{\eta}$ and, 
    as a callable, implements (relying on :class:`BilinearMap` internally)
    the quadratic form
    $$(x, \eta)\mapsto x^{\top} (Q_{\eta} x)$$
    where $Q_\eta = A^{\top}_{\eta} A_{\eta}$.
    
    To construct an instance in terms of a given (symmetric!) $Q$,
    see :func:`Semi2Norm.from_sym_tensorrep`.
    """

    def __init__(self, input_space_trafo_rep: LinearMap) -> None:
        r"""
        Construct An Instance

        Parameters
        ----------
        input_space_trafo_rep : LinearMap
            Corresponds to :math:`A`
        """
        super(QuadraticMap, self).__init__()
        self.input_space_trafo_rep = input_space_trafo_rep
        """ This implements $\eta\mapsto A_\eta$"""

    def forward(self, input, param: Optional[torch.Tensor] = None):
        transformed_input = self.input_space_trafo_rep(input, param)
        return _batch_dot(transformed_input, transformed_input)
    
    @classmethod
    def from_sym_tensorrep(cls, sym_rep: torch.Tensor) -> QuadraticMap:
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
        return QuadraticMap(LinearMap.from_tensorrep(input_space_trafo_rep))


