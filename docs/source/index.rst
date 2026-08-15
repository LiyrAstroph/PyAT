.. PyAT documentation master file, created by
   sphinx-quickstart on Sun Jul 26 17:38:46 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyAT's documentation
=================================

PyAT is a Python package for advanced analysis of astronomical data. It provide 
the following features:

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

Reference: `Li, Y.-R. & Wang, J.-M., 2026, ApJ, submitted <https://ui.adsabs.harvard.edu/abs/2014ApJ...786L...6L/abstract>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started.rst
   ccf.rst