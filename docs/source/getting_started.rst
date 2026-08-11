.. _getting_started:

***************
Getting Started
***************

Requirements
============

PyCALI depends on the following third-party packages:

- numpy
- matplotlib
- astropy
- scipy
- emcee
- celeriate
- corner

These packages are available from the Python package index (PyPI).
One can install them using pip:

.. code-block:: bash

    pip install numpy matplotlib astropy scipy emcee celeriate corner

Installation
============

One can install PyAT from the source code as follows:

.. code-block:: bash

    git clone https://github.com/liyropt/PyAT.git
    cd PyAT
    python -m pip install --no-build-isolation .

After installation, change the directory to **test/** and execute the Python script 
to test the installation.

.. code-block:: python 

    import pyat 

    print(pyat.version())

Quick Start
===========

One can import PyAT in a Python script as follows:

.. code-block:: python

    import pyat

or import PyAT modules, e.g., 

.. code-block:: python 

    from pyat import iccf
