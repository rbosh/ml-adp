.. _installation

Installation
============

Get Python ~= 3.7 and pip-install the repo to your environment :code:`env`.
For example::
    
    (env) ➜ pip install git+https://github.com/rwlmu/ml-adp

To use :code:`ml_adp` in Jupyter notebooks install the IPython kernel dependencies to the environment and create the kernel from within the environment::

    (env) ➜ pip install "ml_adp[jupyter]"
    (env) ➜ python -m ipykernel install --name kernelname

Now, select the kernel :code:`kernelname` in your Jupyter application instance.
