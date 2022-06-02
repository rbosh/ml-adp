r"""Parametrized Mappings

A *parametrized mapping* is a family $(f_{\eta})_{\eta\in\mathbb{R}^p}$ of mappings
$$f_{\eta}\colon \mathbb{R}^k\to \mathbb{R}^m,\quad x\mapsto f_{\eta}(x).$$
Readily speak of $(f_{\eta})$ as if it were a single mapping $\mathbb{R}^k\to \mathbb{R}^m$ to indicate the contextual role of the parameter $\eta$.
Accordingly, expect to find families $(f_{\eta})$ of linear (and related) mappings in :mod:`ml_adp.mapping.linear` and families $(f_{\eta})$ of convex mappings in :mod:`ml_adp.mapping.convex`.
Also, flatten the argument signature of parametrized functions and equivalently consider such $(f_{\eta})$ to be single functions
$$f\colon \mathbb{R}^k\times\mathbb{R}^p\to\mathbb{R}^m, \quad (x,\eta)\mapsto f_{\eta}(x)$$
which is the form the implementations of parametrized mappings in :mod:`ml_adp.mapping` (and its submodules) present themselves in.

In summary, :mod:`ml_adp.mapping` (and its submodules) contain callables with signature ``(input_, param)``, implementing functions $(x,\eta)\mapsto f_{\eta}(x)$ primarily to be considered variates in their first argument and, as such, understood to be parametrized by their second argument.
"""

from . import linear, convex