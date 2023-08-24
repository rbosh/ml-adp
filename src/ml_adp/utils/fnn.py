r"""Basic feedforward neural network components
"""

from __future__ import annotations
import inspect

import itertools as it
import math
from typing import Optional
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch.nn import functional as F


ArraySize = int | Sequence[int]
FNNSize = Sequence[ArraySize]


class BatchNorm(Module):
    r"""Batch norm with generic support for spatial data"""
    def __init__(self,
                 num_features: ArraySize,
                 spatial_dim: int = 0,
                 **kwargs) -> None:
        super().__init__()
        
        if spatial_dim > 3:
            raise ValueError(f"Data dimension {spatial_dim} not supported. Must be <= 3.")

        self.num_features = (num_features,) if isinstance(num_features, int) else num_features
        r"""The feature size of the data"""
        self.spatial_dim = spatial_dim
        r"""The dimension of the spatial domain of the data"""
        
        num_features_flat = np.prod(num_features)
        batch_norm_kwargs = inspect.getfullargspec(torch.nn.BatchNorm1d.__init__)[0]
        batch_norm_kwargs.remove('self')
        batch_norm_config = {key: value for (key, value) in kwargs.items() if key in batch_norm_kwargs}

        batch_norm_dim = spatial_dim if spatial_dim > 0 else 1
        self._batch_norm = getattr(torch.nn, f"BatchNorm{batch_norm_dim}d")(num_features_flat, **batch_norm_config)
        
    def forward(self, input: Tensor) -> Tensor:
        input_flat = input.flatten(start_dim=-(self.spatial_dim + len(self.num_features)), 
                                   end_dim=(feature_axis := -(self.spatial_dim + 1)))
        return self._batch_norm(input_flat).unflatten(feature_axis, self.num_features)
    
    def __repr__(self) -> str:
        return (
            f"BatchNorm({self.num_features}, spatial_dim={self.spatial_dim}, eps={self._batch_norm.eps}, "
            f"momentum={self._batch_norm.momentum}, affine={self._batch_norm.affine}, "
            f"track_running_stats={self._batch_norm.track_running_stats})"
        )


class Linear(Module):
    r"""Linear transformation with (constrained) weight and bias
    
    Given the feature size $n=(n_0,\dots,n_k)$ of input data and an output feature size $m=(m_0,\dots,m_l)$ as well as the dimension $d$ of the data, implements a (local) linear tranformation of the data.
    
    More precisely, saves
    
    * a linear transformation weight matrix $W \in \mathbb{R}^{m\times n}$
    * optionally, a bias term $b \in \mathbb{R}^m$
    * optionally, a *constraint function* $\phi\colon\mathbb{R}\to\mathbb{M}$ with $M$ open in $\mathbb{R}$ (to be applied to $W$ entry-wisely)
    
    and, as a callable, applies
    $$A\colon \mathbb{R}^n\to\mathbb{R}^m,\quad x\mapsto \phi(W)x + b$$
    locally to the input, i.e., point-wisely along spatial axes (as well as batch axes).
    """
    def __init__(self,
                 in_features: ArraySize,
                 out_features: ArraySize,
                 spatial_dim: int = 0,
                 bias: bool = True,
                 constraint_func: Optional[Module | str] = None,
                 uniform_init_range: Optional[tuple[float, float]] = None,
                 **kwargs) -> None:
        r"""Construct a :class:`Linear`-instance
        
        Parameters
        ----------
        in_features
            The feature size of the input
        out_features
            The feature size of the output
        spatial_dim
            The dimension of the spatial domain of the input (default: 0)
        bias
            Indicates inclusion of bias term; default True
        constraint_func
            Use to specify constraint function $\phi$; optional, default None (indicates no use of constraint function)
        uniform_init_range
            The upper and lower bound within which the weight $W$'s and bias $b$'s values are randomly (uniformly and independently) initialized in; optional, default None (indicates *Kaiming* upper and lower bound)
        """
        super().__init__()

        if spatial_dim > 3:
            raise ValueError(f"Data dimension {spatial_dim} not supported. Must be <= 3.")
        
        self.in_features = (in_features,) if isinstance(in_features, int) else in_features
        self.out_features = (out_features,) if isinstance(out_features, int) else out_features
        self.spatial_dim = spatial_dim
        
        in_features_flat = np.prod(in_features)
        out_features_flat = np.prod(out_features)
        
        if constraint_func is not None:
            if isinstance(constraint_func, str):
                constraint_func = getattr(torch.nn, constraint_func)()
            self.add_module('constraint_func', constraint_func)
        else:
            self.register_parameter('constraint_func', None)    
        r"""Weight parametrization $\varphi$; default: identity transformation"""
        
        self.uniform_init_range = uniform_init_range
        """ The range in which to initialize the unconstrained weight"""
        
        self.unconstrained_weight = \
            torch.nn.Parameter(Tensor(out_features_flat, in_features_flat))
        r""" Unconstrained weight representation $W$"""
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features_flat))
            r""" Bias term $b$, optional; default: ``None`` (indicates $b=0$)"""
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.uniform_init_range is not None:
            torch.nn.init.uniform_(self.unconstrained_weight, *self.uniform_init_range)
        else:
            torch.nn.init.kaiming_uniform_(self.unconstrained_weight)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.constraint_func(self.unconstrained_weight) if self.constraint_func is not None else self.unconstrained_weight
                )
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        input_flat = input.flatten(start_dim=-(self.spatial_dim + len(self.in_features)), 
                                   end_dim=(feature_axis := -(self.spatial_dim + 1)))

        output_flat = F.linear(input_flat.transpose(feature_axis, -1),
                               self.constraint_func(self.unconstrained_weight) if self.constraint_func is not None else self.unconstrained_weight,
                               self.bias).transpose(-1, feature_axis)
        
        return output_flat.unflatten(feature_axis, self.out_features)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, spatial_dim={}, bias={}'.format(
            self.in_features, self.out_features, self.spatial_dim, self.bias is not None
        )


class Layer(Sequential):
    r"""Plain neural network layer

	Saves

	    * a :class:`Linear`-instance $A$
        * an activation function $\sigma \colon \mathbb{R} \to \mathbb{R}$
        * optionally, a :class:`BatchNorm`-instance $\langle \cdot \rangle$
    
    and, as a callable, applies
    $$L \colon \mathbb{R}^n \to \mathbb{R}^m, \quad \colon x \mapsto \sigma(A(\langle x\rangle)))$$
    locally to the input.
    """

    def __init__(self,
                 linear: Linear,
                 batch_norm: Optional[BatchNorm] = None,
                 activation: Optional[Module | str] = None) -> None:
        r"""Create a :class:`Layer` as a composite of a :class:`Linear`, a :class:`BatchNorm`, and an activation function
        
        Parameters
        ----------
        linear
            The linearity $A$
        batch_norm
            The batch norm; default None (indicates no batch normalization)
        activation
            the activation function $\sigma$; default None (indicates identity activation function)
        """
        super().__init__()
        
        if batch_norm is not None:
            self.add_module('batch_norm', batch_norm)
        else:
            self.register_parameter('batch_norm', None)
        
        self.add_module('linear', linear)
        
        if activation is not None:
            if isinstance(activation, str):
                activation = getattr(torch.nn, activation)()
            self.add_module('activation', activation)
        else:
            self.register_parameter('activation', None)
     
    @classmethod
    def from_config(cls, 
                    in_features: ArraySize, 
                    out_features: ArraySize, 
                    spatial_dim: int = 0, **layer_config) -> Layer:
        r"""Construct a Layer From Configuration Kwargs

        To configure the layer specify the input feature size $n$ and output feature size $m$, and, additionally,

        * use a Boolean `batch_normalize` to indicate the usage of a batch norm
        * use `activation` to pass the specific activation function (or an identifying `str`).
        
        All other kwargs are passed to the constructors of the constituent parts of the :class:`Layer`, such that for example
        
        * `bias=True` indicates :attr:`linear` to have a non-zero bias term
        * `batch_norm_affine=False` indicates :attr:`batch_norm` to not use the *affine* parameters


        Parameters
        ----------
        in_features
            The input feature size
        out_features
            The output feature size
        spatial_dim
            The dimension of the spatial domain of the features (default: 0)
        **layer_config
            Keyword arguments specifying the configuration of the layer
        """
        
        batch_normalize = layer_config.get('batch_normalize', True)
        if batch_normalize:
            batch_norm = BatchNorm(in_features, spatial_dim=spatial_dim, **layer_config)
        else:
            batch_norm = None

        linear = Linear(in_features, out_features, spatial_dim=spatial_dim, **layer_config)
        
        activation = layer_config.get('activation', None)

        if isinstance(activation, str):
            activation = getattr(torch.nn, activation)()
        
        return Layer(linear, batch_norm=batch_norm, activation=activation)


class FNN(Sequential):
    r"""Plain fully-connected feedforward neural network

    Essentially, a sequence $L_0,\dots, L_N$ of :class:`Layer`'s of compatible feature sizes.
    As a callable, implements their sequential (local) application to the input:
    $$x \mapsto L_N(\dots L_0(x)\dots).$$
    """
    def __init__(self, *layers: Layer) -> None:
        r"""
        Construct an FNN from given layers

        Parameters
        ----------
        *layers
            The layers
        """
        super().__init__(*layers)

    @classmethod
    def from_config(cls, size: FNNSize, spatial_dim: int = 0, **config) -> FNN:
        """Construct an FNN from coniguration kwargs

        To create the :class:`FNN` specify the size of the network (the sequence of the feature sizes of the layers) using ``size`` and additionally configure
        
        * a single activation function to use for all hidden layers using ``hidden_activation``; default ``'LeakyReLU'``
        * a specific activation function for the output layer using ``output_activation``; default `None`
        
        All other kwargs are passed to the constructors of the constituent layers of the :class:`FNN`, which themselves pass them to the constructors of their constituent :class:`Linear`'s and :class:`BatchNorm`'s such that for example
        
        * ``batch_normalize=False`` indicates *all* layers to not use a batch norm
        * ``constraint_func='ReLU'`` indicates *all* layers to have weights constrained by $\operatorname{ReLU}$
        * ``bias=False`` indicates *all* layers to not have a bias term
        * specifying both `activation` and `hidden_activation` generates a warning/exception and gives precedence to `hidden_activation`


        Parameters
        ----------
        size
            The size of the network
        spatial_dim
            The dimension of the spatial domain of the input (default: 0)
        **config
            Keyword arguments specifying the configuration of the network

        Returns
        -------
        FNN
            The configured network
        """

        config = config.copy()

        hidden_activation = config.get('hidden_activation', 'LeakyReLU')
        output_activation = config.get('output_activation', None)
        activations = ([hidden_activation] * (len(size) - 2) + [output_activation])
        
        layers = []
        for i in range(len(size) - 1):
            config['activation'] = activations[i]
            layers.append(Layer.from_config(size[i], size[i + 1], spatial_dim=spatial_dim, **config))

        return FNN(*layers)

    def __add__(self, other: FNN | Layer) -> FNN:
        r"""
        Concatenate :class:`FNN` with other :class:`FNN` (or single :class:`Layer`) on the right

        If the calling :class:`FNN` has the layers :math:`(L_0,\dots, L_N)` and `other` is 
        an :class:`FNN` with layers :math:`(K_0, \dots, K_M)` or is a layer :math:`K`, then the result is the :class:`FNN`
        with layers :math:`(L_0, \dots, L_N, K_0, \dots, K_M)` or :math:`(L_0,\dots, L_N, K)`, respectively.

        Parameters
        ----------
        other
            To be appended on the right

        Returns
        -------
        FNN
            The concatenated FNN

        Raises
        ------
        ValueError
            Raised, if `other` is neither an :class:`FNN` nor a :class:`Layer`
        """
        if isinstance(other, FNN):
            return FNN(*it.chain(self.children(), other.children()))
        elif isinstance(other, Layer):
            other = FNN(other)
            return self + other
        else:
            raise ValueError("Cannot add `other` to `self`.")

