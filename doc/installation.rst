.. _installation

Installation
============

Clone the repo, get Python :code:`~= 3.7` and pip install the repo to your environment.
For the use of :code:`ml_adp` in Jupyter notebooks do (activate the environment first)

.. code ::
    
    $ pip install "ml_adp[jupyter]"
    $ python -m ipykernel install --name kernelname

and select the kernel :code:`kernelname` in your Jupyter application instance.

To locally build the documentation do (from within the :code:`ml_adp` repo root and with the activated environment):

.. code ::

    $ pip install "ml_adp[dev-dependencies]"
    $ cd ./doc
    $ make html