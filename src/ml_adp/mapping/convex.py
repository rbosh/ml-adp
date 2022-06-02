""" Universal Approximators of (Parametrized) Convex Mappings
"""

from typing import Callable, Sequence, Union, Dict, Optional
import torch
from ml_adp.nn import ModuleList, FFN, Layer

SpaceSize = Union[int, Sequence[int]]
FFNSize = Sequence[SpaceSize]

class PICNN(torch.nn.Module):
    r"""Partially Input-Convex Neural Network Architecture (PICNN)
    
    Structurally, the composite of
    
    * an *output net* of size $(k_0,\dots, k_{J+1})$ ultimately producing the output of the architecture and consisting of
            
        * *propagation* :class:`Layer`'s $A_0,\dots, A_J$ of sizes $A_j\colon\mathbb{R}^{k_j}\to\mathbb{R}^{k_{j+1}}$ with a common constraint function $\phi\colon\mathbb{R}\to \mathbb{R}_{\geq 0}$ with non-negative range and no activation functions
        * *residual connection* :class:`Layer`'s $B_0,\dots, B_J$ of sizes $B_j\colon\mathbb{R}^{k_0}\to\mathbb{R}^{k_{j+1}}$ with no activation functions
        * an activation function $\sigma$ to use for the hidden layers as well as an activation function $\rho$ to use for the output layer, both convex and increasing
        
    * a *parameter net*, an :class:`FFN` $L=(L_0,\dots, L_J)$ of size $(p_0,\dots,p_{J+1})$, processing the given parameter
    * *parameter heads*, that is `Layer`'s $U_0,\dots, U_J$ with a common activation function $\psi\colon\mathbb{R}\to\mathbb{R}_{\geq 0}$ with non-negative range (called *floor function*) and layers $V_0,\dots, V_J$, $W_0,\dots, W_J$, with sizes $$U_j\colon\mathbb{R}^{p_j}\to\mathbb{R}^{k_j},\quad V_j\colon\mathbb{R}^{p_j}\to \mathbb{R}^{k_0},\quad W_j\colon\mathbb{R}^{p_j}\to \mathbb{R}^{k_{j+1}}$$ with no activation functions, integrating the parameter net forward propagation results into the output net propagation.
    
    As a callable, implements the function
    $$f\colon (x,\eta)\mapsto y$$
    of input $x$, parameter $\eta$ and of output $y = x_{J+1}$ the final result of the forward propagation of $x_0 = x$ in
    $$x_{j+1} = \sigma(A_j(x_j\odot \eta^{(U)}_j) + B_j(x_0\odot \eta^{(V)}_j) + \eta^{(W)}_j),\quad j=0,\dots, J$$
    where
    $$\eta^{(U)}_j = U_j(\eta_j),\quad \eta^{(V)}_j = V_j(\eta_j),\quad \eta^{(W)}_j = W_j(\eta_j)$$
    and $(\eta_j)_j$ is itself the forward propagation of $\eta_0 = \eta$ in $L$ (meaning $\eta_{j+1} = L_j(\eta_j),\quad j=0,\dots, J$). 
    
    It can be shown that for all parameters $\eta$ the partial function
    $$f_\eta = f(\cdot, \eta)\colon \mathbb{R}^{k_0}\to\mathbb{R}^{k_{J+1}},\quad x\mapsto f(x,\eta)$$
    is convex.
    For this to be the case it is indeed crucial that
    
    * the activation functions of the output net all are convex and increasing (this is the case for most common activation functions including $\mathrm{ReLU}$ and $\mathrm{ELU}$)
    * the constraint functions of the linearities of the layers $A_0,\dots, A_J$ have non-negative range (meaning their transformation weights are constrained to have non-negative entries only, see the documentation of :class:`Linear`) and these layer's batch norms do not have affine parameters
    * the floor function, i.e. the activation function of the parameter head layers $U_0,\dots, U_J$, has non-negative range
        
    Introduced in `Amos et al.`_, the architecture has universal approximation capabilities in the class of families of convex functions.
    
    .. _Amos et al.: https://arxiv.org/abs/1609.07152
    """
    
    def __init__(self,
                 output_net_size: FFNSize,
                 param_net_size: FFNSize,
                 output_net_hidden_activation: Callable = None,
                 output_net_output_activation: Callable = None,
                 floor_func: Optional[Callable] = None,
                 constraint_func: Optional[Callable] = None,
                 propagation_layers_config: Optional[Dict] = None,
                 residual_layers_config: Optional[Dict] = None,
                 param_net_config: Optional[Dict] = None,
                 parameter_heads_config: Optional[Dict] = None) -> None:
        r"""Construct a PICNN Instance
        
        To construct an instance necessarily specify the size $(k_0,\dots, k_{J+1})$ of the output net and the size $(p_0,\dots, p_{J+1})$ of the parameter net and optionally configure the additional parameters.
        
        
        Parameters
        ----------
        output_net_size : FFNSize
            The size $(k_0,\dots, k_{J+1})$ of the output net
        param_net_size : FFNSize
            The size $(p_0,\dots, p_{J+1})$ of the parameter net
        output_net_hidden_activation : Optional[Callable]
            The increasing convex activation function $\sigma$ to be used for the hidden layers of the output net; optional, default ``None`` (indicates usage of $\mathrm{ReLU}$-activation function)
        output_net_output_activation : Optional[Callable]
            The increasing convex activation function $\rho$ to be used for the output layer of the output net; optional, default ``None`` (indicates no activation function)
        floor_func : Optional[Callable]
            The floor function, i.e. the activation function $\psi$ with non-negative range to be used for the parameter head layers $U_0,\dots, U_J$; optional, default ``None`` (indicates $\mathrm{ReLU}$)
        constraint_func : Optional[Callable]
            The constraint function $\phi$ with non-negative range to be used for the output nets propagation layers' linearities; optional, default None (indicates $\exp$)
        propagation_layers_config : Optional[Dict]
            Layer-configuarion to use to construct the propagation layers $A_0,\dots, A_J$; the ``constraint_func``-value will be updated with the given ``constraint_func`` and the ``batch_norm_affine``-value will be updated with ``False``; optional, default ``None``
        residual_layers_config : Optional[Dict]
            Layer-configuarion to use to construct the residual layers $B_0,\dots, B_J$; the ``activation``-value will be updated with the given ``floor_func``, the ``bias``-value will be updated with `False`; optional, default ``None``
        parameter_heads_config : Optional[Dict]
            Layer-configuration to use  to construct the parameter heads $U_0,\dots, U_J$, $V_0,\dots, V_J$, $W_0,\dots, W_J$; for the construction of $U_0,\dots, U_J$, the ``activation``-value will be updated with ``floor_func``, ``bias``-key's value will be updated with ``False``; optional, default ``None``
        param_net_config : Optional[Dict]
            FFN-configuration to use to construct the parameter net, passed directly to the :class:`FFN`-constructor; optional, default ``None`` (indicating not to pass any configuration, leading to the default FFN parameters)
        
        Raises
        ------
        AssertionError
            Raised, if the lengths of `output_net_size` and `param_net_size` differ
        """
        super().__init__()
        
        assert len(output_net_size) == len(param_net_size)
          
        # Output Net Setup
        self.A = torch.nn.ModuleList()
        r"""The sequence of propagation layers $A=(A_0,\dots, A_J)$"""
        self.B = torch.nn.ModuleList()
        r"""The sequence of residual connection layers $B=(B_0,\dots, B_J)$"""
        
        hidden_activation = output_net_hidden_activation if output_net_hidden_activation is not None else torch.nn.ReLU()
        output_activation = output_net_output_activation if output_net_output_activation is not None else torch.nn.Identity()
        self.activations = ModuleList(*([hidden_activation] * (len(output_net_size) - 2) + [output_activation]))
        r"""The sequence of activation functions $(\sigma_0,\dots, \sigma_{J-1}, \sigma_J) = (\sigma,\dots, \sigma, \rho)$"""
        
        propagation_layers_config = {} if propagation_layers_config is None else propagation_layers_config.copy()
        propagation_layers_config.update({
            'constraint_func': torch.exp if constraint_func is None else constraint_func,
            'batch_norm_affine': False
        })
        # propagation_layers_config.update('activation', None) TODO do this?
        # propagation_layers_config.update('bias', False) TODO do this?
        
        residual_layers_config = {} if residual_layers_config is None else residual_layers_config.copy()
        # residual_layers_config.update('activation', None) TODO do this?
        residual_layers_config.update({'bias': False})  # TODO do this?
        
        # Parameter Net Setup
        param_net_config = {} if param_net_config is None else param_net_config
        self.L = FFN.from_config(size=param_net_size, **param_net_config)
        r"""The output net $L=(L_0,\dots, L_J)$"""
        
        # Parameter Heads Setup
        self.U = torch.nn.ModuleList()
        r"""The sequence of parameter head layers $U=(U_0,\dots, U_J)$ with non-negative activation functions"""
        self.V = torch.nn.ModuleList()
        r"""The sequence of parameter head layers $V=(V_0,\dots, V_J)$"""
        self.W = torch.nn.ModuleList()
        r"""The sequence of paramter head layers $W=(W_0,\dots, W_J)$"""

        parameter_heads_config = {} if parameter_heads_config is None else parameter_heads_config.copy()
        #parameter_heads_config.update({'bias': False})
        U_parameter_heads_config = parameter_heads_config.copy()
        U_parameter_heads_config.update({'activation': torch.nn.ReLU() if floor_func is None else floor_func})
        
        for j in range(len(output_net_size) - 1):
            self.A.append(Layer.from_config(
                output_net_size[j],
                output_net_size[j+1],
                **propagation_layers_config
            ))
            self.B.append(Layer.from_config(
                output_net_size[0],
                output_net_size[j+1],
                **residual_layers_config
            ))
            self.U.append(Layer.from_config(
                param_net_size[j],
                output_net_size[j],
                **U_parameter_heads_config
            ))
            self.V.append(Layer.from_config(
                param_net_size[j],
                output_net_size[0],
                **parameter_heads_config
            ))
                #nn.Linear(param_config[i], in_features)
            self.W.append(Layer.from_config(
                param_net_size[j],
                output_net_size[j+1],
                **parameter_heads_config
            ))

    def __len__(self):
        return len(self.L)

    @property
    def output_net_modules(self):
        r"""The Modules Composing the Output Net
        
            It can make sense to optimize the parameters of these modules separately (from the param net modules)
            
            Returns
            -------
            torch.nn.ModuleDict
                Dictionary containing $A$, $B$, and the activation functions
        """
        output_net_keys = ['activations', 'A', 'B']
        return torch.nn.ModuleDict({key: self._modules[key] for key in output_net_keys})

    @property 
    def param_net_modules(self) -> torch.nn.ModuleDict:
        r"""The Modules Composing the Parameter Net and the Parameter Heads
        
            It can make sense to optimize the parameters of these modules separately (from the output net modules)
        
            Returns
            -------
            torch.nn.ModuleDict
                Dictionary containing $L$, $U$, $V$, $W$
        """
        param_net_keys = ['L', 'U', 'V', 'W']
        return torch.nn.ModuleDict({key: self._modules[key] for key in param_net_keys})

    def forward(self, inputs, params):

        intermediates = inputs

        for k in range(len(self)):
            propagation = self.A[k](intermediates * self.U[k](params))
            residual_connection = self.B[k](inputs * self.V[k](params))
            parameter_bias = self.W[k](params)
            intermediates = propagation + residual_connection + parameter_bias
            intermediates = self.activations[k](intermediates)
            params = self.L[k](params)

        return intermediates
    

class PICNN2(torch.nn.Module):
    r"""Partially Input-Convex Neural Network Architecture (PICNN) With Independently Specifiable Output and Parameter Net
    
    .. note::
        This input-convex neural network architecture is a slight modification of :class:`PICNN` introduced by `Amos et al.`_.
        
        Compared to the latter architecture, it allows to different lengths of the output net and parameter net, decoupling the systems and allowing for more flexibility in fine-tuning the architecture to the given task.
        
    .. _Amos et al.: https://arxiv.org/abs/1609.07152
    
    Structurally, the composite of
    
    * an *output net* of size $(k_0,\dots, k_{J+1})$ ultimately producing the output of the architecture and consisting of
            
        * *propagation* :class:`Layer`'s $A_0,\dots, A_J$ of sizes $A_j\colon\mathbb{R}^{k_j}\to\mathbb{R}^{k_{j+1}}$ with a common constraint function $\phi\colon\mathbb{R}\to \mathbb{R}_{\geq 0}$ with non-negative range and no activation functions
        * *residual connection* :class:`Layer`'s $B_0,\dots, B_J$ of sizes $B_j\colon\mathbb{R}^{k_0}\to\mathbb{R}^{k_{j+1}}$ with no activation functions
        * an activation function $\sigma$ to use for the hidden layers as well as an activation function $\rho$ to use for the output layer, both convex and increasing
        
    * a *parameter net*, an :class:`FFN` $L=(L_0,\dots, L_I)$ of size $(p_0,\dots,p_{I+1})$, processing the given parameter
    * *parameter heads*, that is `Layer`'s $U_0,\dots, U_J$ with a common activation function $\psi\colon\mathbb{R}\to\mathbb{R}_{\geq 0}$ with non-negative range (called *floor function*) and layers $V_0,\dots, V_J$, $W_0,\dots, W_J$, with sizes $$U_j\colon\mathbb{R}^{p_{I+1}}\to\mathbb{R}^{k_j},\quad V_j\colon\mathbb{R}^{p_{I+1}}\to \mathbb{R}^{k_0},\quad W_j\colon\mathbb{R}^{p_{I+1}}\to \mathbb{R}^{k_{j+1}}$$ with no activation functions, integrating the parameter net forward propagation results into the output net propagation.
    
    As a callable, implements the function
    $$f\colon (x,\eta)\mapsto y$$
    of input $x$, parameter $\eta$ and of output $y = x_{J+1}$ the final result of the forward propagation of $x_0 = x$ in
    $$x_{j+1} = \sigma(A_j(x_j\odot \eta^{(U)}_j) + B_j(x_0\odot \eta^{(V)}_j) + \eta^{(W)}_j),\quad j=0,\dots, J$$
    where
    $$\eta^{(U)}_j = U_j(\eta_{I+1}),\quad \eta^{(V)}_j = V_j(\eta_{I+1}),\quad \eta^{(W)}_j = W_j(\eta_{I+1})$$
    and $\eta_{I+1}$ is the final result of the forward propagation of $\eta_0 = \eta$ in $L$ ($\eta_{i+1} = L_i(\eta_i),\quad i=0,\dots, I$). 
    
    It can be shown that for all parameters $\eta$ the partial function
    $$f_\eta = f(\cdot, \eta)\colon \mathbb{R}^{k_0}\to\mathbb{R}^{k_{J+1}},\quad x\mapsto f(x,\eta)$$
    is convex.
    For this to be the case it is indeed crucial that
    
    * the activation functions of the output net all are convex and increasing (this is the case for most common activation functions including $\mathrm{ReLU}$ and $\mathrm{ELU}$)
    * the constraint functions of the linearities of the layers $A_0,\dots, A_J$ have non-negative range (meaning their transformation weights are constrained to have non-negative entries only, see the documentation of :class:`Linear`) and these layer's batch norms do not have affine parameters
    * the floor function, i.e. the activation function of the parameter head layers $U_0,\dots, U_J$, has non-negative range
        
    Introduced in `Amos et al.`_, the architecture has universal approximation capabilities in the class of families of convex functions.
    
    .. _Amos et al.: https://arxiv.org/abs/1609.07152
    """
    
    def __init__(self,
                 output_net_size: FFNSize,
                 param_net_size: FFNSize,
                 output_net_hidden_activation: Callable = None,
                 output_net_output_activation: Callable = None,
                 floor_func: Optional[Callable] = None,
                 constraint_func: Optional[Callable] = None,
                 propagation_layers_config: Optional[Dict] = None,
                 residual_layers_config: Optional[Dict] = None,
                 param_net_config: Optional[Dict] = None,
                 parameter_heads_config: Optional[Dict] = None) -> None:
        r"""Construct a PICNN Instance
        
        To construct an instance necessarily specify the size $(k_0,\dots, k_{J+1})$ of the output net and the size $(p_0,\dots, p_{I+1})$ of the parameter net and optionally configure the additional parameters.
        
        
        Parameters
        ----------
        output_net_size : FFNSize
            The size $(k_0,\dots, k_{J+1})$ of the output net
        param_net_size : FFNSize
            The size $(p_0,\dots, p_{I+1})$ of the parameter net
        output_net_hidden_activation : Optional[Callable]
            The increasing convex activation function $\sigma$ to be used for the hidden layers of the output net; optional, default ``None`` (indicates usage of $\mathrm{ReLU}$-activation function)
        output_net_output_activation : Optional[Callable]
            The increasing convex activation function $\rho$ to be used for the output layer of the output net; optional, default ``None`` (indicates no activation function)
        floor_func : Optional[Callable]
            The floor function, i.e. the activation function $\psi$ with non-negative range to be used for the parameter head layers $U_0,\dots, U_J$; optional, default ``None`` (indicates $\mathrm{ReLU}$)
        constraint_func : Optional[Callable]
            The constraint function $\phi$ with non-negative range to be used for the output nets propagation layers' linearities; optional, default None (indicates $\exp$)
        propagation_layers_config : Optional[Dict]
            Layer-configuarion to use to construct the propagation layers $A_0,\dots, A_J$; the ``constraint_func``-value will be updated with the given ``constraint_func`` and the ``batch_norm_affine``-value will be updated with ``False``; optional, default ``None``
        residual_layers_config : Optional[Dict]
            Layer-configuarion to use to construct the residual layers $B_0,\dots, B_J$; the ``activation``-value will be updated with the given ``floor_func``, the ``bias``-value will be updated with `False`; optional, default ``None``
        parameter_heads_config : Optional[Dict]
            Layer-configuration to use  to construct the parameter heads $U_0,\dots, U_J$, $V_0,\dots, V_J$, $W_0,\dots, W_J$; for the construction of $U_0,\dots, U_J$, the ``activation``-value will be updated with ``floor_func``, ``bias``-key's value will be updated with ``False``; optional, default ``None``
        param_net_config : Optional[Dict]
            FFN-configuration to use to construct the parameter net, passed directly to the :class:`FFN`-constructor; optional, default ``None`` (indicating not to pass any configuration, leading to the default FFN parameters)
        """
        super().__init__()
        
        # Output Net Setup
        self.A = torch.nn.ModuleList()
        r"""The sequence of propagation layers $A=(A_0,\dots, A_J)$"""
        self.B = torch.nn.ModuleList()
        r"""The sequence of residual connection layers $B=(B_0,\dots, B_J)$"""
        
        hidden_activation = output_net_hidden_activation if output_net_hidden_activation is not None else torch.nn.ReLU()
        output_activation = output_net_output_activation if output_net_output_activation is not None else torch.nn.Identity()
        self.activations = ModuleList(*([hidden_activation] * (len(output_net_size) - 2) + [output_activation]))
        r"""The sequence of activation functions $(\sigma_0,\dots, \sigma_{J-1}, \sigma_J) = (\sigma,\dots, \sigma, \rho)$"""
        
        propagation_layers_config = {} if propagation_layers_config is None else propagation_layers_config.copy()
        propagation_layers_config.update({
            'constraint_func': torch.exp if constraint_func is None else constraint_func,
            'batch_norm_affine': False
        })
        # propagation_layers_config.update('activation', None) TODO do this?
        # propagation_layers_config.update('bias', False) TODO do this?
        
        residual_layers_config = {} if residual_layers_config is None else residual_layers_config.copy()
        # residual_layers_config.update('activation', None) TODO do this?
        residual_layers_config.update({'bias': False})  # TODO do this?
        
        # Parameter Net Setup
        param_net_config = {} if param_net_config is None else param_net_config
        self.L = FFN.from_config(size=param_net_size, **param_net_config)
        r"""The output net $L=(L_0,\dots, L_I)$"""
        
        # Parameter Heads Setup
        self.U = torch.nn.ModuleList()
        r"""The sequence of parameter head layers $U=(U_0,\dots, U_J)$ with non-negative activation functions"""
        self.V = torch.nn.ModuleList()
        r"""The sequence of parameter head layers $V=(V_0,\dots, V_J)$"""
        self.W = torch.nn.ModuleList()
        r"""The sequence of paramter head layers $W=(W_0,\dots, W_J)$"""

        parameter_heads_config = {} if parameter_heads_config is None else parameter_heads_config.copy()
        #parameter_heads_config.update({'bias': False})
        U_parameter_heads_config = parameter_heads_config.copy()
        U_parameter_heads_config.update({'activation': torch.nn.ReLU() if floor_func is None else floor_func})
        
        for j in range(len(output_net_size) - 1):
            self.A.append(Layer.from_config(
                output_net_size[j],
                output_net_size[j+1],
                **propagation_layers_config
            ))
            self.B.append(Layer.from_config(
                output_net_size[0],
                output_net_size[j+1],
                **residual_layers_config
            ))
            self.U.append(Layer.from_config(
                param_net_size[-1],
                output_net_size[j],
                **U_parameter_heads_config
            ))
            self.V.append(Layer.from_config(
                param_net_size[-1],
                output_net_size[0],
                **parameter_heads_config
            ))
                #nn.Linear(param_config[i], in_features)
            self.W.append(Layer.from_config(
                param_net_size[-1],
                output_net_size[j+1],
                **parameter_heads_config
            ))

    def __len__(self):
        return len(self.L)

    @property
    def output_net_modules(self):
        r"""The Modules Composing the Output Net
        
            It can make sense to optimize the parameters of these modules separately (from the param net modules)
            
            Returns
            -------
            torch.nn.ModuleDict
                Dictionary containing $A$, $B$, and the activation functions
        """
        output_net_keys = ['activations', 'A', 'B']
        return torch.nn.ModuleDict({key: self._modules[key] for key in output_net_keys})

    @property 
    def param_net_modules(self) -> torch.nn.ModuleDict:
        r"""The Modules Composing the Parameter Net and the Parameter Heads
        
            It can make sense to optimize the parameters of these modules separately (from the output net modules)
        
            Returns
            -------
            torch.nn.ModuleDict
                Dictionary containing $L$, $U$, $V$, $W$
        """
        param_net_keys = ['L', 'U', 'V', 'W']
        return torch.nn.ModuleDict({key: self._modules[key] for key in param_net_keys})

    def forward(self, input, param):

        param = self.L(param)

        for k in range(len(self)):
            propagation = self.A[k](input * self.U[k](param))
            residual_connection = self.B[k](input * self.V[k](param))
            parameter_bias = self.W[k](param)
            input = propagation + residual_connection + parameter_bias
            input = self.activations[k](input)

        return input
    
