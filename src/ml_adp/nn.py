"""
Provides Basic Neural Network Components
"""
from __future__ import annotations
import inspect

import itertools as it
import math
from contextlib import contextmanager
from typing import Optional, Sequence, Union, Tuple, Any, Callable
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F

SpaceSize = Union[int, Sequence[int]]
FFNSize = Sequence[SpaceSize]


class BatchNorm(torch.nn.Module):
    def __init__(self,
                 num_features: SpaceSize,
                 **batch_norm_config):
        super().__init__()
        self.num_features = (num_features,) if isinstance(num_features, int) else num_features
        r""" Input Features Size"""
        
        num_features_flat = np.prod(num_features)
        batch_norm_kwargs = inspect.getfullargspec(torch.nn.BatchNorm1d.__init__)[0]
        batch_norm_kwargs.remove('self')
        batch_norm_config = {key: value for (key, value) in batch_norm_config.items() if key in batch_norm_kwargs}
        self._batch_norm1d = torch.nn.BatchNorm1d(num_features_flat, **batch_norm_config)
        r""" Underlying One-Dimensional Batch Norm Module"""
        
    def forward(self, input: torch.Tensor):
        return self._batch_norm1d(input.flatten(start_dim=1)).view((-1,) + self.num_features)
    
    def __repr__(self):
        return (
            f"BatchNorm({self.num_features}, eps={self._batch_norm1d.eps}, "
            f"momentum={self._batch_norm1d.momentum}, affine={self._batch_norm1d.affine}, "
            f"track_running_stats={self._batch_norm1d.track_running_stats})"
        )


class Linear(torch.nn.Module):
    r""" Linear Transformation with Parametrized Weight and Optional Bias
    
    Given a :attr:`constraint_func` $\varphi\colon \mathbb{R}\to O$ with $O\subseteq \mathbb{R}$, implements
    $$x\mapsto \varphi(W)x + b$$
    where $W\in\mathbb{R}^{m\times n}$ is initialized with values in :attr:`uniform_init_range`
    and $\varphi(W)\in\mathbb{R}^{m\times n}$ is thus constrained to have entries in $O$.
    """
    def __init__(self,
                 in_features: SpaceSize,
                 out_features: SpaceSize,
                 bias: bool = True,
                 constraint_func: Optional[Callable] = None,
                 uniform_init_range: Optional[Sequence[float]] = None,
                 **_config_dump):
        super().__init__()
        
        self.in_features = (in_features,) if isinstance(in_features, int) else in_features
        self.out_features = (out_features,) if isinstance(out_features, int) else out_features
        
        in_features_flat = np.prod(in_features)
        out_features_flat = np.prod(out_features)
        
        if constraint_func is None:
            constraint_func = torch.nn.Identity()
        object.__setattr__(self, 'constraint_func', constraint_func)
        r""" Weight parametrization $\varphi$; default: identity transformation"""
        
        self.uniform_init_range = uniform_init_range
        """ Indicate the range in which to initialize the unconstrained weight"""
        
        self.unconstrained_weight = \
            torch.nn.Parameter(torch.Tensor(out_features_flat, in_features_flat))
        r""" Unconstrained weight representation $W$"""
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features_flat))
            r""" Bias term $b$, optional; default: ``None`` (indicates $b=0$)"""
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

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

    def forward(self, input: torch.Tensor):
        # TODO It is a bug that the case out_features = (0, 0) is not supported, this is because of Tensor.view not being
        # able to infer ( `t.view(-1, 0, 0)` )the first dimension in that case
        # This all the while works: torch.rand(50, 0, 0).flatten(start_dim=1).view(50, 0, 0)
        return F.linear(input.flatten(start_dim=1),
                        self.constraint_func(self.unconstrained_weight),
                        self.bias).view((-1,) + self.out_features)
        
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, constaint_func={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.constraint_func
        )


class Layer(torch.nn.Sequential):
    r"""
    Plain Neural Network Layer Architecture

    As a callable, implements
    $$x\mapsto \sigma(A\langle x\rangle))$$
    where $\langle \cdot\rangle$ is either a batch norm or a no-op (depending on specification),
    $A\colon x\mapsto Wx + b$ is an affine map with weight $W$ and bias $b$,
    and $\sigma$ is a specified activation function.
    """

    def __init__(self,
                 linear: Linear,
                 batch_norm: BatchNorm = None,
                 activation: Callable = None):
        super().__init__()
        
        if batch_norm is not None:
            self.add_module('batch_norm', batch_norm)
        else:
            self.register_parameter('batch_norm', None)
        
        self.add_module('linear', linear)
        
        if activation is not None:
            self.add_module('activation', activation)
        else:
            self.register_parameter('activation', None)
     
    @classmethod
    def from_config(cls, in_features: SpaceSize, out_features: SpaceSize, **layer_config) -> Layer:
        r"""Construct a Layer with Certain Configuration.

        To configure the layer

        * to use a specific activation function, specify it as a kwarg `activation`, default: `None`
        * to not have a bias, i.e. $b=0$, specify `bias=False`
        * to not have a batch norm, specify `batch_norm=False`
        * to have the batch norm not be *affine*, specify `batch_norm_affine=False`
        * to use a certain constraint function for the linear transformation weight, specify it in `linear_constraint_func`


        Parameters
        ----------
        in_features : int
            The dimension of the space of the input data
        out_features : int
            The dimension of the space of the output data
        **config
            Keyword arguments specifying the configuration of the layer
        """
        
        batch_normalize = layer_config.get('batch_normalize', True)
        if batch_normalize:
            batch_norm = BatchNorm(in_features, **layer_config)
        else:
            batch_norm = None
        linear = Linear(in_features, out_features, **layer_config)
        activation = layer_config.get('activation', None)
        
        return Layer(linear, batch_norm=batch_norm, activation=activation)


class FFN(torch.nn.Sequential):
    r"""
    Plain Fully-Connected Feed-Forward Neural Network Architecture

    Essentially, a sequence $L_0,\dots, L_N$ of :class:`Layer`'s of compatible feature sizes,
    that, as a callable, implements their sequential application:
    $$x \mapsto L_N(\dots L_0(x)\dots).$$
    """
    def __init__(self, *layers: Layer) -> None:
        r"""
        Construct an FFN From Given Layers

        Parameters
        ----------
        *layers: Layer
            The layers
        """
        super().__init__(*layers)

    @classmethod
    def from_config(cls, sizes: FFNSize, **config) -> FFN:
        """
        Construct an FFN of Certain Configuration

        `sizes` determines the number of neurons of the layers.
        To additionally control the FFN's layers, specify a shared layer configuration 
        by using keyword arguments (which, internally, are passed to :meth:`Layer.from_config`).
        Exception, do not specify ``activation``, rather
        * specify ``hidden_activation``, the activation function for the hidden layers; default: :class:`torch.nn.ReLU`
        * specify ``output_activation``, the activation function for the output layer; default: `None`

        Parameters
        ----------
        sizes : FFNSize
            The sizes of the layers
        **config: 
            Keyword arguments for layer configuration

        Returns
        -------
        FFN
            The configured network
        """

        config = config.copy()

        hidden_activation = config.get('hidden_activation', torch.nn.ELU())
        output_activation = config.get('output_activation', None)
        activations = ([hidden_activation] * (len(sizes) - 2)
                       + [output_activation])
        
        layers = []
        for i in range(len(sizes) - 1):
            config['activation'] = activations[i]
            layers.append(Layer(sizes[i], sizes[i+1], **config))

        return FFN(*layers)

    def __add__(self, other: Union[FFN, Layer]) -> FFN:
        r"""
        Concatenate :class:`FFN` with other :class:`FFN` (or single :class:`Layer`) on the right

        If the calling :class:`FFN` has the layers :math:`(L_0,\dots, L_N)` and `other` is 
        an :class:`FFN` with layers :math:`(K_0, \dots, K_M)` or is a layer :math:`K`, then the result is the :class:`FFN`
        with layers :math:`(L_0, \dots, L_N, K_0, \dots, K_M)` or :math:`(L_0,\dots, L_N, K)`, respectively.

        Parameters
        ----------
        other : Union[FFN, Layer]
            To be appended on the right

        Returns
        -------
        FFN
            The concatenated FFN

        Raises
        ------
        ValueError
            Raised, if `other` is neither an :class:`FFN` nor a :class:`Layer`
        """
        if isinstance(other, FFN):
            return FFN(*it.chain(self.children(), other.children()))
        elif isinstance(other, Layer):
            other = FFN(other)
            return self + other
        else:
            raise ValueError("Cannot add `other` to `self`.")


class IPReLU(torch.nn.Module):
    """ Increasing PReLU Activation Function
    """
   
    #__constants__ = ['num_parameters']
    #num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 #constraint_func = None,
                 device=None, dtype=None) -> None:
        super(IPReLU, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        #if constraint_func is None:
        self.constraint_func = torch.nn.Sigmoid()
        self.unconstrained_weight = torch.nn.Parameter(torch.empty(num_parameters, **factory_kwargs).fill_(init))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.prelu(input, self.constraint_func(self.unconstrained_weight))

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


class MultiHead(torch.nn.ModuleList):
    """ Apply Multiple Modules in Parallel
    
    Saves :class:`torch.nn.Module`'s $N_0,\dots, N_K$, and, as a callable,
    implements
    $$ x \mapsto (N_0(x), \dots, N_K(x))$$

    A *head* $N_j$ being ``None`` corresponds to the $j$-th entry of the output being ``None``.
    """
    def __init__(self, *heads: Optional[torch.nn.Module]) -> None:
        """
        Construct Given a Sequence of Heads

        Parameters
        ----------
        *heads: torch.nn.Module
            The sequence of heads $N_0,\dots, N_K$
        """
        super().__init__(heads)

    def forward(self, input_: torch.Tensor) -> Tuple[Optional[torch.Tensor]]:
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