# ml-adp

`ml_adp` is a little Python utility embedding into the Pytorch neural network and optimization framework and serves 
for the numerical solution of discrete-time finite-horizon stochastic optimal control problems.
It exports a list-like interface to the central functional components of such optimal control problems allowing for concise implementations of numerical methods that rely on the approximate satisfaction of the Bellman equations (Approximate Dynamic Programming; ADP).


## Installation

Clone the repo, get Python ~= 3.7 and `poetry` and do (from within the repo root)
```
$ poetry install
```
For use of `ml_adp` in Jupyter notebooks do
```
$ poetry install --extras jupyter
$ poetry run python -m ipykernel install --name kernelname
```
and select the kernel `kernelname` in your Jupyter instance.

## Documentation

See link.

