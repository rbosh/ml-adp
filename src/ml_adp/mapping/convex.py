""" Universal Approximators of (Parametrized) Convex Mappings
"""

from typing import Callable, Sequence, Union, Dict, Optional
import torch
from ml_adp.nn import ModuleList, FFN, Layer

SpaceSize = Union[int, Sequence[int]]
FFNSize = Sequence[SpaceSize]

class PICNN(torch.nn.Module):
    r"""Partially Input-Convex Neural Network Architecture (PICNN)
    
    Structurally, a composite of
    
    * an *output net* of size $(k_0,\dots, k_{J+1})$        
        * activation functions $g_0,\dots, g_J$, 
        * propagation :class:`Layer`s $A_0,\dots, A_J$ of sizes $A_j\colon\mathbb{R}^{k_j}\to\mathbb{R}^{k_{j+1}}$ with a common constraint function $\phi\mathbb{R}\to M$ with non-negative range $M$
        * residual connection :class:`Layer`'s $B_0,\dots, B_J$ with sizes $B_j\mathbb{R}^{k_0}\to\mathbb{R}^{k_{j+1}}$
        
    and ultimately producing the output of the architecture
    * a *parameter net*, an :class:`FFN` $L=(L_0,\dots, L_J)$ of size $(p_0,\dots,p_{J+1})$, processing the parameter value
    * *parameter heads*, $\ReLU$-:class:`Layer`'s $U_0,\dots, U_J$ and $\id_\R$-layers $V_0,\dots, V_J$, $W_0,\dots, W_J$, with sizes
    $$U_j\colon\reals{p_j}\to\reals{k_j},\quad V_j\colon\reals{p_j}\to \reals{k_0},\quad W_j\colon\reals{p_j}\to \reals{k_{j+1}}$$ 
    and no biases, integrating the parameter net forward propagation results into the output net propagation
    
    As a callable, implements
    $$f\colon (u,\eta)\mapsto v$$
    where $v = u_{J+1}$ is the final result produced by the forward propagation $u_0 = u$
    $$u_{j+1} = g_j(A_j(u_j\odot \eta^{(U)}_j) + B_j(u_0\odot \eta^{(V)}_j) + \eta^{(W)}_j),\quad j=0,\dots, J$$
    in which
    $$\eta^{(U)}_j = U_j(\eta_j),\quad \eta^{(V)}_j = V_j(\eta_j),\quad \eta^{(W)}_j = W_j(\eta_j)$$
    and $(\eta_j)_j$ is itself the forward propagation of $\eta$ in $L$, meaning $\eta_0 = \eta$ and $\eta_{j+1} = L_j(\eta_j)$ for $j=0,\dots, J$. 
    
    It can be shown that for all parameters $\eta$ the function
    $$f_\eta = f(\var, \eta)\colon \mathbb{R}^{k_0}\to\mathbb{R}^{k_{J+1}}, u\mapsto f(u,\eta)$$
    is convex, provided that
    
    * the activation functions of the output net are convex and increasing (this is the case for most common activation functions including $\mathrm{ReLU}$ and $\mathrm{ELU}$)
    * the constraint functions of linearities of the layers $A_0,\dots, A_J$ have non-negative range (meaning their transformation weights are constrained to have non-negative entries only, see the documentation of :class:`Linear`)
    * the activation functions of the parameter head layers $U_0,\dots, U_J$ have non-negative range
    * no affine batch norms
    
    which as conditions might as well be included into the definition of PICNN's.
    
    Introduced in `Amos et al.`_, the architecture can be expected to have satisfying approximation capabilties.
    
    .. _Amos et al.: https://arxiv.org/abs/1609.07152
    """
    
    
    def __init__(self,
                 output_net_size: FFNSize,
                 param_net_size: FFNSize,
                 output_net_hidden_activation: Callable = None,
                 output_net_output_activation: Callable = None,
                 floor_func: Optional[Callable] = None,
                 constraint_func: Optional[Callable] = None,
                 param_net_config: Optional[Dict] = None,
                 residual_layers_config: Optional[Dict] = None,
                 propagation_layers_config: Optional[Dict] = None,
                 parameter_heads_config: Optional[Dict] = None) -> None:
        r"""Construct a PICNN Instance
        
        To construct an instance specify the size $(k_0,\dots, k_{J+1})$ of the output net and the size $(p_0,\dots, p_{J+1})$ of the parameter net using `output_net_size` and `param_net_size`, respectively.
        
        
        Parameters
        ----------
        output_net_size : FFNSize
            The size $(k_0,\dots, k_{J+1})$ of the output net
        param_net_size : FFNSize
            The size $(p_0,\dots, p_{J+1})$ of the parameter net
        output_net_hidden_activation : Optional[Callable]
            The increasing convex activation function to be used for the hidden layers of the output net; optional, default None (indicates usage of $\mathrm{ReLU}$)
        output_net_output_activation : Optional[Callable]
            The increasing convex activation function to be used for the output layer of the output net; optional, default None (indicates no activation function)
        floor_func : Optional[Callable]
            The activation function with non-negative range to be used for the parameter head layers $U_0,\dots, U_J$; optional, default None (indicates $\mathrm{ReLU}$)
        constraint_func : Optional[Callable]
            The constraint function with non-negative range to be used for the output nets propagation layers's linearities; optional, default None (indicates $\exp$)
        param_net_config : Optional[Dict]
            FFN-configuration dict of the parameter net, passed to the :class:`FFN`-constructor of the parameter net; optional, default None (indicating not to pass any configuration leading to the default parameters)
        residual_layers_config : Optional[Dict]
            Layer-configuarion dict to use to construct the residual layers $B_0,\dots, B_J$, `activation`-key's value will be updated with `floor_func`, `bias`-key's value will be updated with `False`; optional, default None
        propagation_layers_config : Optional[Dict]
            Layer-configuarion dict to use to construct the propagation layers $A_0,\dots, A_J$, `constraint_func`-key's value will be updated with `constraint_func` and `batch_norm_affine` will be updated with `False`; optional, default None
        parameter_heads_config : Optional[Dict]
            Layer-configuration dict to be used to construct the parameter heads $U_0,\dots, U_J$, $V_0,\dots, V_J$, $W_0,\dots, W_J$; for the construction of $U_0,\dots, U_J$, the `activation`-key's value will be updated with `floor_func`, `bias`-key's value will be updated with `False`; optional, default None
        
        Raises
        ------
        AssertionError
            Raised, if the lengths of `output_net_size` and `param_net_size` differ
        """
        super().__init__()
        
        assert len(output_net_size) == len(param_net_size)
          
        # Output Net Setup
        self.A = torch.nn.ModuleList()
        self.B = torch.nn.ModuleList()
        
        hidden_activation = output_net_hidden_activation if output_net_hidden_activation is not None else torch.nn.ReLU()
        output_activation = output_net_output_activation if output_net_output_activation is not None else torch.nn.Identity()
        self.activations = ModuleList(*([hidden_activation] * (len(output_net_size) - 2) + [output_activation]))
        
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
        self.L = FFN.from_config(sizes=param_net_size, **param_net_config)
        
        # Parameter Heads Setup
        self.U = torch.nn.ModuleList()
        self.V = torch.nn.ModuleList()
        self.W = torch.nn.ModuleList()

        parameter_heads_config = {} if parameter_heads_config is None else parameter_heads_config.copy()
        parameter_heads_config.update({'bias': False})
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
    
