PyAT
=====

**Py**\ thon **A**\ stronomical **T**\ ools.

PyAT is a package providing useful tools for astronomical analysis.
It is in continuous development and currently provides
the following procedures

- Estimate the systematic error of a light curve;
- Generate formatted input files for MICA package;
- Calculate mean and rms spectra and their line widths;
- Remove outliers from a light curve using the median filter;
- Rebin spectra by keeping the total flux unchanged;
- Calculate interpolated cross-correlation function (ICCF);
- Reconstruct light curves using damped random walk model;
- Detrend light curves using a linear trend;
- AGN spectral templates (Glikman et al. 2006; Vanden Berk et al. 2001);
- Quasar spectral energy distribution templates (Elvis et al. 1994; Shang et al. 2011);
- Merge spectral segments;
- Smooth data using Savitzky-Golay filter;
- Cosmological calculations.

To install PyAT, use the terminal command 

.. code-block:: python
    
    python setup.py install --user
    # or
    python -m pip install .
  