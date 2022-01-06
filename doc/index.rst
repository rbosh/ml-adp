.. ml_adp documentation master file, created by
   sphinx-quickstart on Mon Sep 13 23:10:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


ml-adp
======

``ml_adp`` is a Python package embedding into the Pytorch neural network and optimization framework and serves 
the numerical solution of discrete-time finite-horizon stochastic optimal control problems.
It exports a list-like interface to the central functional components of such optimal control problems, allowing for concise implementations of numerical methods that rely on the approximate satisfaction of the Bellman equations (Approximate Dynamic Programming; ADP).

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   guide
   examples/index


Reference
---------

.. autosummary::
   :toctree: _autosummary
   :template: custom_module.rst
   :caption: Reference
   :recursive:

   ml_adp


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
