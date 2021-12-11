"""
Provides Parametrized Convex Mappings
"""

import ml_adp.mapping.linear
import torch
from ml_adp.nn import Linear


class PICNN1(torch.nn.Module):
    r"""
    Partially Input-Convex Neural Network Architecture

    Implements the map $(x, y)\mapsto x_{N+1}$ with $x_{N+1}$ as in
	$$x_{k+1} = g_k\left(A_k(x_k\cdot u_k) + B_k(x_0\cdot v_k) + w_k\right),\quad k=0,\dots, N$$
    where
    
    * $L=(L_0,\dots, L_N)$ an :class:`FFN` with size $(m_0, \dots, m_N)$ and $(y_k)_k$ the forward propagation of $y_0 = y$ in $L$,

    * for all $k=0,\dots, N$ $U_k$ a $ReLU$-:class:`Layer` of size $(m_k, n_k)$ and $u_k = U_k(y_k)$, $V_k$ a $id_{\mathbb{R}}$-layer of size $(m_k, n_0)$ and $v_k = V_k(y_k)$, and $W_k$ a $id_\mathbb{R}$-layer of size $(m_k, n_{k+1})$ and $w_k = W_k(y_k)$, 

    * and for all $k=0,\dots, N$ $A_k$ an $id_{\mathbb{R}}$-layer of size $(n_k, n_{k+1})$ and with a weight matrix with non-negative entries, $B_k$ an $id_{\mathbb{R}}$-layer of size $(n_0, n_{k+1})$, and $g_k$ a convex and increasing activation function, and $x_0 = x$
    """

    def __init__(self,
                 input_config: list,
                 param_config: list,
                 hidden_activation=None,
                 output_activation=None,
                 param_hidden_activation=None,
                 param_output_activation=None,
                 floor_func=None):
        super(PICNN1, self).__init__()

        if hidden_activation is None:
            hidden_activation = torch.nn.ELU()
        if param_hidden_activation is None:
            param_hidden_activation = torch.nn.ELU()
        if output_activation is None:
            output_activation = torch.nn.Identity()
        if floor_func is None:
            floor_func = torch.nn.ReLU()

        self.activations = [hidden_activation] * (len(input_config) - 2) + [output_activation]
        self.floor_func = floor_func

        self.input_norm = torch.nn.BatchNorm1d(input_config[0], affine=False)

        self.L = ml_adp.nn.FFN.from_config(
            param_config,
            hidden_activation=param_hidden_activation,
            output_activation=param_output_activation,
            batch_normalize=False
        )

        self.param_norms = torch.nn.ModuleList()
        self.A = torch.nn.ModuleList()
        self.U = torch.nn.ModuleList()
        self.B = torch.nn.ModuleList()
        self.V = torch.nn.ModuleList()
        self.W = torch.nn.ModuleList()

        for i in range(len(input_config) - 1):
            self.param_norms.append(torch.nn.BatchNorm1d(param_config[i]))
            self.A.append(ml_adp.nn.Layer(
                input_config[i],
                input_config[i+1],
                activation=None,
                bias=False,
                constraint_func=torch.exp
            ))
            self.U.append(ml_adp.nn.Layer(
                param_config[i],
                input_config[i],
                activation=self.floor_func,
                batch_normalize=False
            ))
            self.B.append(ml_adp.nn.Layer(
                input_config[0],
                input_config[i+1],
                activation=None,
                bias=False
            ))
            self.V.append(ml_adp.nn.Layer(
                param_config[i],
                input_config[0],
                activation=None,
                batch_normalize=False
            ))
                #nn.Linear(param_config[i], in_features)
            self.W.append(ml_adp.nn.Layer(
                param_config[i],
                input_config[i+1],
                activation=None,
                batch_normalize=False
            ))
                #nn.Linear(param_config[i], input_config[i+1])

    def __len__(self):
        return len(self.L)

    def forward(self, inputs, params):

        inputs_normed = self.input_norm(inputs)
        # params = self.param_norm(params)
        intermediates = inputs
        
        for k in range(len(self)):
            intermediates1 = self.A[k](intermediates * self.U[k](params))
            intermediates2 = self.B[k](inputs * self.V[k](params))  # Maybe use inputs normed here
            intermediates = intermediates1 + intermediates2 + self.W[k](params)
            intermediates = self.activations[k](intermediates)
            params = self.L[k](params)

        return intermediates
