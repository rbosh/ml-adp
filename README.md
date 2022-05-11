# ml-adp

`ml_adp` is a Python package embedding into the Pytorch neural network and optimization framework and serves 
the numerical solution of discrete-time finite-horizon stochastic optimal control problems.
It exports a list-like interface to the central functional components of such optimal control problems, allowing for concise implementations of numerical methods that rely on the approximate satisfaction of the discrete-time Bellman equations (Approximate Dynamic Programming; ADP).

## Installation

Get Python ~= 3.7 and pip-install the repo to your environment `env`.
For example: 
```
(env) ➜ pip install git+https://github.com/rwlmu/ml-adp
```

To use `ml_adp` in Jupyter notebooks install the IPython kernel dependencies to the environment and create the kernel from within the environment
```
(env) ➜ pip install "ml_adp[jupyter]"
(env) ➜ python -m ipykernel install --name kernelname
```
Now, select the kernel `kernelname` in your Jupyter application instance.

## Documentation

Documentation is available [here](https://ml-adp.readthedocs.io/en/latest/).
