.. _ccf_label:

Interpolated Cross Correlation Function
=======================================

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

   r = \frac{\sum_{i=1}^{n} (x'_i-\bar x')(y'_i-\bar y')}{\sqrt{\sum_{i=1}^n(x'_i-\bar x')^2\sum_{i=1}^{n}(y'_i-\bar y')^2}},

where $n$ is the number of points in the light curves,

.. math::

   \bar x' = \frac{1}{n}\sum_{i=1}^{n}x'_i,

and

.. math::

   \bar y' = \frac{1}{n}\sum_{i=1}^{n}y'_i.

The linear interpolation is used in calculating ICCF.