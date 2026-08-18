.. _getting_started:

***************
Getting Started
***************

Requirements
============

PyAT depends on the following third-party packages:

- `Numpy <https://numpy.org/>`_
- `Matplotlib <https://matplotlib.org/>`_
- `Astropy <https://www.astropy.org/>`_
- `Scipy <https://scipy.org/>`_
- `Emcee <https://emcee.readthedocs.io/en/stable/>`_
- `Celerite <https://celerite.readthedocs.io/en/stable/>`_
- `Corner <https://corner.readthedocs.io/en/latest/>`_

These packages are available from the Python package index (PyPI).
One can install them using pip:

.. code-block:: bash

    pip install numpy matplotlib astropy scipy emcee celerite corner

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

To check the modules and functions in PyAT, one can use the built-in function **dir()**:

.. code-block:: python

    import pyat
    print(dir(pyat))

The output looks like this::

    ['__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', 
     '__name__', '__package__', '__path__', '__spec__', '__version__', '_version', 
     'ccf', 'ccf_null_test', 'ccf_old', 'cosmology', 'detrend', 'drw', 'drw_modeling', 
     'drw_modeling_fast', 'drw_recon', 'estimate_syserr', 'estimate_syserr_median_filter', 
     'filter', 'format_mica', 'genlc_psd_drw', 'genlc_psd_drw_data', 'genlc_psd_pow', 
     'get_bin_edge', 'get_line_widths', 'get_mean_rms', 'iccf', 'iccf_mc', 'iccf_mc_oneway', 
     'iccf_mc_oneway_slow', 'iccf_mc_slow', 'iccf_ndeff', 'iccf_oneway', 
     'iccf_oneway_peak_significance', 'iccf_oneway_slow', 'iccf_peak', 'iccf_peak_oneway', 
     'iccf_peak_significance', 'iccf_prmax_null', 'iccf_sigma_null', 'iccf_slow', 
     'list_seds', 'list_templates', 'load_mica', 'load_sed', 'load_template', 'loadsed', 
     'loadtemplate', 'mean_rms_spectra', 'rebin', 'rebin_sig', 'rebin_spectrum', 
     'rebin_spectrum_with_error', 'remove_outliers', 'smooth_savgol', 'spec_merge', 
     'version']

To check only the modules in PyAT, one can use the following command:

.. code-block:: python

    import pyat
    for mod in pkgutil.iter_modules(pyat.__path__):
        print(mod.name)

The output looks like this::

    _version
    ccf
    ccf_null_test
    ccf_old
    cosmology
    detrend
    drw
    estimate_syserr
    filter
    format_mica
    loadsed
    loadtemplate
    mean_rms_spectra
    rebin
    rebin_spectrum
    remove_outliers
    sim_rm
    spec_merge


Tests
=====

In the subfold **test/**, there are some example scripts to demonstrate the usage of PyAT.