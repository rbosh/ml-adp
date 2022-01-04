# ml-adp

`ml_adp` is a Python package embedding into the Pytorch neural network and optimization framework and serves 
the numerical solution of discrete-time finite-horizon stochastic optimal control problems.
It exports a list-like interface to the central functional components of such optimal control problems, allowing for concise implementations of numerical methods that rely on the approximate satisfaction of the discrete-time Bellman equations (Approximate Dynamic Programming; ADP).

## Installation

Clone the repo, get Python ~= 3.7 and pip-install the repo to your environment.
For using `ml_adp` in Jupyter notebooks do (activate the environment first)
```    
$ pip install "ml_adp[jupyter]"
$ python -m ipykernel install --name kernelname
```
and select the kernel `kernelname` in your Jupyter application instance.


## Documentation

Documentation is available [here](https://ml-adp.readthedocs.io/en/latest/).
