"""
Provides Basic Neural Network Components
"""
from __future__ import annotations

import itertools as it
import math
from contextlib import contextmanager
from typing import Optional, Sequence, Union, Tuple, Any, Callable
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F

SpaceSize = Union[int, Sequence[int]]
FFNDims = Sequence[SpaceSize]


class InView(torch.nn.Module):
    """
    As callable, flattens all :class:`torch.Tensor` input
    """
    # TODO Avoid inherting torch.nn.Module, for this need torch.nn.Sequential to allow non torch.nn.Modules
    def __init__(self) -> None:
        """
        Construct an :class:`InView` object
        """
        super(InView, self).__init__()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Flatten multi-dimensional input `input_`

        Parameters
        ----------
        input_ : torch.Tensor
            The potentially multi-dimensional input

        Returns
        -------
        torch.Tensor
            The flattened input
        """
        return input_.flatten(start_dim=1)


class OutView(torch.nn.Module):
    """
    As a callable, rearranges input to have different size/dimensionality

    See documentation of torch.view
    """
    def __init__(self, view_size: SpaceSize) -> None:
        """
        Construct an `OutView` instance

        Parameters
        ----------
        view_size : SpaceSize
            The new size
        """
        super(OutView, self).__init__()
        self.view_size = torch.Size(view_size)

    def __repr__(self) -> str:
        return (type(self).__name__
                + "("
                + self.view_size.__repr__()
                + ")")

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Rearrange input `input_`

        Parameters
        ----------
        input_ : torch.Tensor
            The input

        Returns
        -------
        torch.Tensor
            The rearranged input
        """
        return input_.view((-1,) + self.view_size)


class Linear(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 constraint_func: Optional[Callable] = None,
                 uniform_init_range: Optional[Sequence[float]] = None):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if constraint_func is None:
            constraint_func = torch.nn.Identity()
        self.constraint_func = constraint_func
        self.uniform_init_range = uniform_init_range
        self.unconstrained_weight = \
            torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # TODO Rewrite the following to not make compositions explode
    def reset_parameters(self):
        if self.uniform_init_range is not None:
            torch.nn.init.uniform_(self.unconstrained_weight, *self.uniform_init_range)
        else:
            torch.nn.init.kaiming_uniform_(self.unconstrained_weight)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.constraint_func(self.unconstrained_weight)
                )
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input,
                        self.constraint_func(self.unconstrained_weight),
                        self.bias)


class Layer(torch.nn.Sequential):
    r"""
    Plain Layer Architecture for Neural Networks

    As a callable, implements
    $$x\mapsto \sigma(A\langle x\rangle))$$
    for batches of input data,
    where $\langle \cdot\rangle$ is either a batch norm or a no-op (depending on specification),
    $A\colon x\mapsto Wx + b$ is an affine map with weight $W$ and bias $b$,
    and $\sigma$ is a specified activation function.
    """

    def __init__(self,
                 in_features: SpaceSize,
                 out_features: SpaceSize,
                 **config) -> None:
        r"""
        Construct a Layer of certain configuration.

        To configure the layer

        * to use a specific activation function, specify it in `activation`, default: `None`
        * to not have a bias, i.e. $b=0$, specify `bias=False`
        * to not have a batch norm, specify `batch_norm=False`
        * to have the batch norm not be *affine*, specify `batch_norm_affine=False`
        * to use a certain constraint function for the weight, specify it in `linear_constraint_func`


        Parameters
        ----------
        in_features : int
            The dimension of the space of the input data
        out_features : int
            The dimension of the space of the output data
        **config
            Keyword arguments specifying the configuration of the layer
        """
        super(Layer, self).__init__()

        in_features_flat = np.prod(in_features)
        out_features_flat = np.prod(out_features)

        activation = config.get('activation', None)
        bias = config.get('bias', True)
        batch_normalize = config.get('batch_normalize', True)
        batch_norm_affine = config.get('batch_norm_affine', True)
        constraint_func = config.get('constraint_func', None)

        if isinstance(in_features, (tuple, list, torch.Size)):
            self.add_module('in_view', InView())
        else:
            self.register_parameter('in_view', None)

        if batch_normalize and in_features_flat != 0:
            self.add_module(
                'batch_norm',
                torch.nn.BatchNorm1d(in_features_flat, affine=batch_norm_affine)
            )
        else:
            self.register_parameter('batch_norm', None)

        self.add_module(
            'linear',
            Linear(in_features_flat,
                   out_features_flat,
                   bias=bias,
                   constraint_func=constraint_func)
        )

        if activation is not None:  # Activations Work for out_features_flat==0
            self.add_module(
                'activation',
                activation
            )
        else:
            self.register_parameter('activation', None)

        if isinstance(out_features, (tuple, list, torch.Size)) and out_features_flat != 0:
            # TODO It is a bug that the case out_features = (0, 0) is not supported, this is because of Tensor.view not being
            # able to infer ( `t.view(-1, 0, 0)` )the first dimension in that case
            # This all the while works: torch.rand(50, 0, 0).flatten(start_dim=1).view(50, 0, 0)
            self.add_module('out_view', OutView(out_features))
        else:
            self.register_parameter('out_view', None)


class FFN(torch.nn.Sequential):
    r"""
    Plain Fully-Connected Feed-Forward Neural Network Architecture

    Essentially, a sequence $L_0,\dots, L_N$ of :class:`Layer`'s of compatible feature sizes,
    and, as a callable, implementing their sequential application:
    $$x \mapsto L_N(\dots(L_0(x)\dots).$$
    """
    def __init__(self, *layers: Layer) -> None:
        r"""
        Construct a FFN from given layers

        Parameters
        ----------
        *layers
            In expanded form, the aribtrary number of layers in order
        """
        super().__init__(*layers)

    @classmethod
    def from_config(cls,
                    sizes: FFNDims,
                    **config) -> FFN:
        """
        Construct a FFN of certain configuration

        To configure the FFN, specify its `size` and configure the consituting :class:`Layers`'s by specifying keyword arguments as described in :func:`nn.Layer.__init__`
        Moreover,
        * to have certain activation function for the output layer, specify `output_activation`, default: `None`
        * to have a certain activation function for all hidden layers, specify `hidden_activation`, default `torch.nn.ELU()`
        * to have a certain sequence of activation functions, specify it as a list as `activations`, default `None`

        Parameters
        ----------
        dims : FFNDims
            The sizes of the layers
        config: 
            Keyword arguments specifying the configuration of the layer

        Returns
        -------
        FFN
            The configured :class:`FFN` network
        """

        hidden_activation = config.get('hidden_activation', torch.nn.ELU())
        output_activation = config.get('output_activation', None)

        """
        Maybe do something linke this
        defaults = {'startDate'       : startDate,
            'endDate'                 : endDate,
            'periodicityAdjustment'   : 'ACTUAL',
            'periodicitySelection'    : 'DAILY',
            'nonTradingDayFillOption' : 'ACTIVE_DAYS_ONLY',
            'adjustmentNormal'        : False,
            'adjustmentAbnormal'      : False,
            'adjustmentSplit'         : True,
            'adjustmentFollowDPDF'    : False}   
        defaults.update(kwargs)
        """ 
        

        default_activations = ([hidden_activation] * (len(sizes) - 2)
                       + [output_activation])
        
        # TODO Warn user if too many kwargs were given
        activations = config.get('activations', default_activations)

        layers = []
        for i in range(len(sizes) - 1):
            config['activation'] = activations[i]
            layers.append(Layer(sizes[i], sizes[i+1], **config))

        return FFN(*layers)

    def __add__(self, other: Union[FFN, Layer]) -> FFN:
        r"""
        Concatenate calling :class:`FFN` with other :class:`FFN` (or wrapped :class:`Layer` on the right)

        If the calling :class:`FFN` has the layers :math:`(L_0,\dots, L_N)` and `other` is 
        the :class:`FFN` with layers :math:`(K_0, \dots, K_M)` or is the layer :math:`K`, then the result is the :class:`FFN`
        :math:`(L_0, \dots, L_N, K_0, \dots, K_M)` or :math:`(L_0,\dots, L_N, K)`, respectively.

        Parameters
        ----------
        other : Union[FFN, Layer]
            The other FFN or layer to append on the right

        Returns
        -------
        FFN
            The concatenated FFN

        Raises
        ------
        ValueError
            Raised, if `other` is not an FFN or a layer
        """
        if isinstance(other, FFN):
            return FFN(*it.chain(self.children(), other.children()))
        elif isinstance(other, Layer):
            other = FFN(other)
            return self + other
        else:
            raise ValueError("Cannot add `other` to `self`.")


class MultiHead(torch.nn.ModuleList):
    def __init__(self, *heads: Optional[torch.nn.Module]) -> None:
        super().__init__(heads)

    def forward(self, input_: torch.Tensor) -> Tuple[Optional[torch.Tensor]]:
        # map(Layer.forward, self, it.repeat(_input))
        #output = []
        #for layer in self:
        #    output.append(None if layer is None else layer(input))
        #return output

        # If head is None just return None at that place
        return tuple(map(lambda head: head and head.forward(input_), self))


class ModuleList(torch.nn.Module):
    def __init__(self, *entries) -> None:
        super().__init__()

        self._non_module_entries = OrderedDict(zip(map(str, range(len(entries))), [None] * len(entries)))
        self.__setitem__(slice(None, None), entries)

    def __repr__(self) -> str:
        return repr([entry for entry in self])

    def _register_entry(self, idx: int, value: Any):
        del self._entry_dict_by_idx(idx)[str(idx)]
        if isinstance(value, torch.torch.nn.Module):
            setattr(self, str(idx), value)
        else:
            self._non_module_entries[str(idx)] = value

    def _entry_dict_by_idx(self, idx: int) -> OrderedDict:
        try:
            self._modules[str(idx)]
        except KeyError:
            return self._non_module_entries
        else:
            return self._modules

    def __len__(self) -> int:
        return len(self._non_module_entries) + len(self._modules)

    def __setitem__(self,
                    key: Union[int, slice],
                    value: Optional[Union[Sequence[Optional[Any]], Callable]]) -> None:

        idx = list(range(len(self)))[key]
        if isinstance(idx, int):
            self._register_entry(idx, value)
        else:
            if callable(value) or value is None:
                value = [value] * len(idx)
            value = list(value)
            assert len(value) == len(idx)
            for i, idx in enumerate(idx):
                self._register_entry(idx, value[i])

    def __getitem__(self,
                    key: Union[int, slice]) -> Union[Optional[Any], ModuleList]:
        idx = list(range(len(self)))[key]
        if isinstance(idx, int):
            return self._entry_dict_by_idx(idx)[str(idx)]
        else:
            return self.__class__(*[self._entry_dict_by_idx(idx)[str(idx)] for idx in idx])


@contextmanager
def _evaluating(model: torch.nn.Module):
    '''
    Temporarily switch to evaluation mode.
    From: https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-
    manager/18998/3
    (MIT Licensed)
    '''
    training = model.training
    try:
        model.eval()
        yield model
    except AttributeError:
        yield model
    finally:
        if training:
            model.train()