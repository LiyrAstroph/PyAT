.. _ccf_label:

Interpolated Cross Correlation Function
=======================================

Mathematical Definition
-----------------------

The cross-correlation function between two light curves, say, :math:`x(t)` and :math:`y(t)`, is defined as

.. math::

    r(\tau) = \frac{E\{[x(t)-\bar{x}][y(t+\tau)-\bar{y}]\}}
    {\sqrt{E\{[x(t)-\bar{x}]^2\}E\{[y(t+\tau)-\bar{y}]^2\}}},

where :math:`\tau` is the time lag, :math:`\bar{x}` and :math:`\bar{y}` are the mean values 
of :math:`x(t)` and :math:`y(t+\tau)`, respectively, and :math:`E\{x\}` is the expected value of :math:`x(t)`.

The realistic light curves in AGN monitoring are usually irregularly sampled and their observing cadences
are also not contemporaneous. Direct calculation of the CCF for AGN light curves is not straightforward. 
Interpolated cross-correlation function (ICCF) is then introduced to cope with irregular sampling.

For the convention, two rounds of cross-correlation coefficients are computed, with only one light curve 
being interpolated in each round. The final cross-correlation coefficient is
then assigned as the average of the two rounds.  Specifically, given a time lag :math:`\tau`, 
firstly shift :math:`x(t)` in time with :math:`\tau` and extract the segment in :math:`x(t+\tau)`
overlapping with :math:`y(t)`, which we denote :math:`x'(t)`. Then interpolate :math:`y(t)` onto the time 
points same as :math:`x'(t)`.
As such, we obtain two light curves :math:`x'(t)` and :math:`y'(t)` with the same duration and sampling rate.
We then directly compute their cross-correlation coefficient as below.
The second round repeats the same procedure but shifts :math:`y(t)` in time with :math:`\tau` and implements 
interpolation on :math:`x(t)`.


After interpolation, the cross-correlation coefficient of the light-curve pairs :math:`x'(t)` and :math:`y'(t)` 
is then calculated as 

.. math::

   r(\tau) = \frac{\sum_{i=1}^{n} (x'_i-\bar x')(y'_i-\bar y')}{\sqrt{\sum_{i=1}^n(x'_i-\bar x')^2\sum_{i=1}^{n}(y'_i-\bar y')^2}},

where $n$ is the number of points in the light curves,

.. math::

   \bar x' = \frac{1}{n}\sum_{i=1}^{n}x'_i,

and

.. math::

   \bar y' = \frac{1}{n}\sum_{i=1}^{n}y'_i.

The linear interpolation is used in calculating ICCF.


PyAT Implementation
---------------------

PyAT provides the following functions to calculate ICCF:

.. function:: iccf(t1, f1, t2, f2, ntau, tau_beg, tau_end, threshold=0.8, mode="multiple", ignore_warnings=False, ways=0)

    :synopsis: Calculate interpolated cross-correlation function (ICCF) between two light curves and determine the peak coefficient and time lag and centroid time lag.

    :param t1: Time array of the first light curve.
    :param f1: Flux array of the first light curve.
    :param t2: Time array of the second light curve.
    :param f2: Flux array of the second light curve.
    :param ntau: Number of time-lag bins to calculate the CCF.
    :param tau_beg: Beginning time lag to calculate the CCF.
    :param tau_end: End time lag to calculate the CCF.
    :param threshold: Threshold to filter out the CCF.
    :param mode: Mode to calculate the CCF, "multiple" or "single".
    :param ignore_warnings: Whether to ignore warnings.
    :param ways: Ways to calculate the CCF, 0: two ways; 1: only interpolate the first light curve; 2: only interpolate the second light curve.

    :return: tau, ccf, rmax, tau_peak, tau_cent
    :rtype: numpy.ndarray, numpy.ndarray, float, float, float


.. function:: iccf_mc(t1, f1, e1, t2, f2, e2, ntau, tau_beg, tau_end, nsim=1000, threshold=0.8, mode="multiple", ignore_warnings=False, ways=0)

    :synopsis: Monte Carlo simulation of interpolated cross-correlation function (ICCF) between two light curves.

    :param t1: Time array of the first light curve.
    :param f1: Flux array of the first light curve.
    :param e1: Error array of the first light curve.
    :param t2: Time array of the second light curve.
    :param f2: Flux array of the second light curve.
    :param e2: Error array of the second light curve.
    :param ntau: Number of time-lag bins to calculate the CCF.
    :param tau_beg: Beginning time lag to calculate the CCF.
    :param tau_end: End time lag to calculate the CCF.
    :param nsim: Number of Monte Carlo simulations to calculate the CCF, default is 1000.
    :param threshold: Threshold to filter out the CCF, default is 0.8.
    :param mode: Mode to calculate the CCF, "multiple" or "single".
    :param ignore_warnings: Whether to ignore warnings.
    :param ways: Ways to calculate the CCF, 0: two ways; 1: only interpolate the first light curve; 2: only interpolate the second light curve.

    :return: ccf_peak_mc, tau_peak_mc, tau_cent_mc
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray

.. function:: iccf_peak(t1, f1, t2, f2, ntau, tau_beg, tau_end)
    
    :synopsis: Calculate interpolated cross-correlation function (ICCF) between two light curves and determine the peak coefficient and time lag.

    :param t1: Time array of the first light curve.
    :param f1: Flux array of the first light curve.
    :param t2: Time array of the second light curve.
    :param f2: Flux array of the second light curve.
    :param ntau: Number of time-lag bins to calculate the CCF.
    :param tau_beg: Beginning time lag to calculate the CCF.
    :param tau_end: End time lag to calculate the CCF.
    :return: tau, ccf, rmax, tau_peak
    :rtype: numpy.ndarray, numpy.ndarray, float, float

Examples
--------
First import PyAT in a Python script as follows:

.. code-block:: python

    import pyat

Then load the light curve data, e.g., take two light curves with a file name of "lc1.txt" and "lc2.txt":

.. code-block:: python

    import numpy as np

    lc1 = np.loadtxt("lc1.txt")
    lc2 = np.loadtxt("lc2.txt")

Here, the file contains three columns, namely time, flux, and error. 
Now calculate the ICCF between the two light curves:

.. code-block:: python
    
    ntau = 1001
    tau_beg = -50.0
    tau_end = 100.0
    threshold = 0.8
    mode = "multiple"
    ignore_warnings = False

    # calculate iccf
    tau, ccf, rmax, tau_peak, tau_cent = pyat.iccf(lc1[:, 0], lc1[:, 1], lc2[:, 0], lc2[:, 1], 
     ntau, tau_beg, tau_end, threshold=threshold, mode=mode, ignore_warnings=ignore_warnings)
    
    # peroform Monte Carlo simulation to determine the time lag uncertainties
    nsim = 1000
    ccf_peak_mc, tau_peak_mc, tau_cent_mc = pyat.iccf_mc(lc1[:, 0], lc1[:, 1], lc1[:, 2], 
    lc2[:, 0], lc2[:, 1], lc2[:, 2], ntau, tau_beg, tau_end, nsim=nsim, threshold=threshold, 
    mode=mode, ignore_warnings=ignore_warnings)

Now plot the results and generate figures.

.. code-block:: python

    import matplotlib.pyplot as plt

    # plot the ICCF
    plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(311)
    plt.plot(tau, ccf, label="ICCF")
    ax.axvline(x=tau_peak, color="red", label="Peak", ls='--')
    ax.axvline(x=tau_cent, color="blue", label="Centroid", ls='--')
    ax.set_xlabel("Time Lag (days)")
    ax.set_ylabel("ICCF")
    ax.legend()
    
    # plot histogram of ICCF peaks from Monte Carlo simulations
    ax = fig.add_subplot(312)
    ax.hist(ccf_peak_mc)
    ax.set_ylabel("Count")
    ax.set_xlabel("ICCF Peak")
    
    # plot histogram of time lags from Monte Carlo simulations
    ax = fig.add_subplot(313)
    ax.hist(tau_peak_mc, label="Peak")
    ax.hist(tau_cent_mc, label="Centroid", alpha=0.5)
    ax.legend()
    ax.set_ylabel("Count")
    ax.set_xlabel("Time Lag (days)")
    plt.show()   
