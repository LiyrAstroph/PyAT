#===================================================================================#
#  PyAT: Python Astronomical Tools
#  A package providing basic, common tools in astronomical analysis
#
#  Yan-Rong Li, liyropt@gmail.com
#  2023-08-31
#===================================================================================#

__all__ = ["iccf_ndeff", "iccf_sigma_null", "iccf_prmax_null"]

import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp

from .ccf import iccf
from .drw import drw_modeling

def _ax_forward(x):
    return np.tanh(x) 

def _ax_inverse(x):
    return np.arctanh(np.clip(x, -0.999, 0.999))

def _ndeff_estimate(n, dt, taux, tauy, errx, sigx, erry, sigy):
    """
    Estimate the effective number of independent samples for a time series.

    This approximates the reduction in the number of independent points caused
    by correlated variability and measurement noise. The estimate is based on
    the local sampling cadence ``dt`` and the characteristic correlation time
    scales ``taux`` and ``tauy`` for the two series, with the noise-to-signal
    terms weighted by ``errx/sigx`` and ``erry/sigy``.

    Parameters
    ----------
    n : int
        Number of data points in the time series.
    dt : float
        Average time spacing between points in the series.
    taux, tauy : float
        Characteristic correlation timescales for the two series.
    errx, erry : float
        Average measurement uncertainties for the two series.
    sigx, sigy : float
        Characteristic variability amplitudes (or noise scales) for the two series.
    
    Returns
    -------
    ndeff : float
        Estimated effective number of independent points in the time series.
        
    """
    sum = 0.0
    for i in range(1, n):
        sum += (1.0 - i/n) * np.exp(-dt*i/taux)/(1.0+errx**2/sigx**2) * np.exp(-dt*i/tauy)/(1.0+erry**2/sigy**2)
    return n/(1.0 + 2.0*sum)

def iccf_ndeff(t1, y1, ye1, t2, y2, ye2, ntau, tau_beg, tau_end, 
               sig1, taud1, sig2, taud2, gapx=None, gapy=None):
    """
    Estimate the number of effective points in each time-lag bin for an
    interpolated cross-correlation function (ICCF) calculation.

    Parameters
    ----------
    t1, y1, ye1 : array-like
        Time vector and measurements with uncertainties for the first light curve.
    t2, y2, ye2 : array-like
        Time vector and measurements with uncertainties for the second light curve.
    ntau : int
        Number of lag bins between tau_beg and tau_end.
    tau_beg, tau_end : float
        Start and end values of the lag grid.
    sig1, sig2 : float
        Intrinsic variability amplitudes (or characteristic noise scales) for the
        two light curves.
    taud1, taud2 : float
        Correlation timescales used in the effective-sample-size estimate for the
        two light curves.
    gapx, gapy : array-like or None, optional
        Gaps in the time sampling for the two light curves, given as ``(start,
        end)`` pairs. They are used to correct the effective temporal spacing when
        estimating the number of independent points.

    Returns
    -------
    tau : ndarray
        Lag values used for the cross-correlation bins.
    nd1, ndeff1 : ndarray
        Number of data points and effective number of independent points in each
        lag bin for the first-direction interpolation.
    nd2, ndeff2 : ndarray
        Number of data points and effective number of independent points in each
        lag bin for the second-direction interpolation.

    Notes
    -----
    For each lag value, the function determines the subset of points in the
    second light curve that overlap the first one after shifting by ``tau`` and
    computes the effective number of independent points using the local sampling
    cadence and the variability/noise parameters. The same is then repeated for
    the inverse lag direction.
    """

    tau = np.linspace(tau_beg, tau_end, ntau)
    
    nd1 = np.zeros(ntau)
    nd2 = np.zeros(ntau)
    ndeff1 = np.zeros(ntau)
    ndeff2 = np.zeros(ntau)
    
    gapx_dur = 0.0
    len_gapx = 0
    if gapx is not None:
        len_gapx = len(gapx)
        for i in range(len(gapx)):
            gapx_dur += gapx[i][1]-gapx[i][0]
    gapy_dur = 0.0
    len_gapy = 0
    if gapy is not None:
        len_gapy = len(gapy)
        for i in range(len(gapy)):
            gapy_dur += gapy[i][1]-gapy[i][0]

    for i in range(ntau):
        taui = tau[i]
        
        # first interpolate y1
        idx = np.where((t2-taui>=t1[0])&(t2-taui<=t1[-1]))[0]
        t2_new = t2[idx]
        # y2_new = y2[idx]
        ye2_new = ye2[idx]
        # y1_new = np.interp(t2_new, t1, y1)
        ye1_new = np.interp(t2_new, t1, ye1)
        nd1[i] = t2_new.shape[0]
        
        err1 = np.mean(ye1_new)
        err2 = np.mean(ye2_new)
        dt = (t2_new[-1]-t2_new[0]-gapx_dur)/(t2_new.shape[0]-1-len_gapx)
        ndeff1[i] = _ndeff_estimate(t2_new.shape[0], dt, taud1, taud2, err1, sig1, err2, sig2)

        # then interpolat y2
        idx = np.where((t1+taui>=t2[0])&(t1+taui<=t2[-1]))[0]
        t1_new = t1[idx]
        # y1_new = y1[idx]
        ye1_new = ye1[idx]
        # y2_new = np.interp(t1_new, t2, y2)
        ye2_new = np.interp(t1_new, t2, ye2)
        nd2[i] = t1_new.shape[0]

        err1 = np.mean(ye1_new)
        err2 = np.mean(ye2_new)
        
        dt = (t1_new[-1]-t1_new[0]-gapy_dur)/(t1_new.shape[0]-1-len_gapy)
        ndeff2[i] = _ndeff_estimate(t1_new.shape[0], dt, taud1, taud2, err1, sig1, err2, sig2)

        # print(taui, y1_new.shape[0], y2_new.shape[0])

    return tau, nd1, ndeff1, nd2, ndeff2

def iccf_sigma_null(t1, y1, ye1, t2, y2, ye2, ntau, tau_beg, tau_end,
                    gapx=None, gapy=None, doplot=False):
    """
    Estimate the null hypothesis standard deviation of the ICCF.

    Parameters
    ----------
    t1, y1, ye1 : array-like
        Time vector and measurements with uncertainties for the first light curve.
    t2, y2, ye2 : array-like
        Time vector and measurements with uncertainties for the second light curve.
    ntau : int
        Number of lag bins between tau_beg and tau_end.
    tau_beg, tau_end : float
        Start and end values of the lag grid.
    gapx, gapy : array-like or None, optional
        Gaps in the time sampling for the two light curves, given as ``(start,
        end)`` pairs. They are used to correct the effective temporal spacing when
        estimating the number of independent points.
    doplot : bool, optional
        If True, plot the results.

    Returns
    -------
    tau : ndarray
        Lag values used for the cross-correlation bins.
    sigma_null : ndarray
        Estimated standard deviation of the ICCF under the null hypothesis, based
        on the effective number of independent points in each lag bin.
    """

    # calculate iccf
    tau, ccf, rmax, tau_peak, tau_cent = iccf(t1, y1, t2, y2, ntau, tau_beg, tau_end)
    print("rmax: %.2f, zmax: %.2f at tau=%.2f"%(rmax, zmax,tau_peak))

    # DRW fits 
    sample1 = drw_modeling(t1, y1, ye1, doshow=True)
    sigma1, tau1 = np.exp(np.median(sample1, axis=0))
    print("DRW sigma and tau: %.2f, %.2f"%(sigma1, tau1))

    sample2 = drw_modeling(t2, y2, ye2, doshow=True)
    sigma2, tau2 = np.exp(np.median(sample2, axis=0))
    print("DRW sigma and tau: %.2f, %.2f"%(sigma2, tau2))

    tau, nd1, ndeff1, nd2, ndeff2 =  iccf_ndeff(t1, y1, ye1, t2, y2, ye2, ntau, tau_beg, tau_end, 
                                                sigma1, tau1, sigma2, tau2, gapx=gapx, gapy=gapy)

    ndeff = 2/(1/ndeff1+1/ndeff2)
    sigma_null = 1/np.sqrt(ndeff)

    # calculate the probability of rmax in null hypothesis
    taudxy = tau1 * tau2/(tau1 + tau2)
    sig = (np.mean(1.0/ndeff))**0.5
    A = 1.0
    nm = (tau_end-tau_beg)/taudxy/2
    nm = np.max((1, nm))
    print("m of p(rmax): %.2f, taudxy: %.2f"%(nm, taudxy))
    m = np.linspace(0-sig*3, 0+sig*5, 1000)
    fm = A * nm/sig * (0.5*(1+sp.erf(m/(np.sqrt(2)*sig))))**(nm-1)*np.exp(-0.5*m**2/sig**2)/np.sqrt(2*np.pi)/sig
    idx = (m >= np.arctanh(rmax)) 
    print("probability of rmax in null hypothesis:", np.sum(fm[idx])/np.sum(fm))

    if doplot:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_axes((0.1, 0.1, 0.4, 0.8))
        y1_mean = np.mean(y1)
        y2_mean = np.mean(y2)
        ax.errorbar(t1, y1/y1_mean+0.5, yerr=ye1/y1_mean, ls='none', elinewidth=0.8, 
                    marker='o', markersize=2, ecolor='grey', capsize=0.8)
        ax.errorbar(t2, y2/y2_mean, yerr=ye2/y2_mean, ls='none', elinewidth=0.8, 
                    marker='o', markersize=2, ecolor='grey', capsize=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux + Offset (arbitrary unit)")
        ax.minorticks_on()
        
        ax = fig.add_axes((0.6, 0.1, 0.35, 0.8))
        plt.plot(tau, ccf)
        ax.set_xlim(tau[0], tau[-1])
        ax.set_xlabel("Time Lag (day)")
        ax.set_ylabel(r"ICCF $(r)$")
        ax.minorticks_on()
        ax.yaxis.set_label_coords(-0.1, 0.5)

        xccf = np.linspace(tau_beg, tau_end, 200)
        yccf = np.linspace(-0.999, 0.999, 200)
        mx, my = np.meshgrid(xccf, yccf)
        sig_ccf = 1/np.interp(xccf, tau, ndeff)
        mz = np.exp(-0.5*np.arctanh(my)**2/(sig_ccf[np.newaxis, :]))
        ax.imshow(mz, aspect="auto", extent=[xccf[0], xccf[-1], yccf[0], yccf[-1]], 
                interpolation="gaussian", zorder=0, cmap="Wistia")
        ax.plot(tau, np.tanh(1.0/ndeff**0.5), ls='-', lw=1, color='gray', label=r'1$\sigma$')
        ax.plot(tau, -np.tanh(1.0/ndeff**0.5), ls='-', lw=1, color='gray')
        ax.plot(tau, np.tanh(2.0/ndeff**0.5), ls='--', lw=1, color='gray', label=r'2$\sigma$')
        ax.plot(tau, -np.tanh(2.0/ndeff**0.5), ls='--', lw=1, color='gray')
        ax.plot(tau, np.tanh(3.0/ndeff**0.5), ls=':', lw=1, color='gray', label=r'3$\sigma$')
        ax.plot(tau, -np.tanh(3.0/ndeff**0.5), ls=':', lw=1, color='gray')
        ax.legend(ncols=3, frameon=True, loc='lower center', columnspacing=1.0, 
                handlelength=1.5, handletextpad=0.4)

        ax2 = ax.twinx()
        ax2.set_ylabel(r"ICCF $(z)$")
        ax_ylim = ax.get_ylim()
        ax2.set_ylim(_ax_inverse(ax_ylim[0]), _ax_inverse(ax_ylim[1]))
        ax2.set_yscale("function", functions=(_ax_forward, _ax_inverse))

        plt.show()

    return tau, sigma_null

def iccf_prmax_null(t1, y1, ye1, t2, y2, ye2, ntau, tau_beg, tau_end,
                   gapx=None, gapy=None, doplot=False):
    """
    Estimate the null hypothesis probability of the ICCF rmax.
    
    Parameters
    ----------
    t1, y1, ye1 : array-like
        Time vector and measurements with uncertainties for the first light curve.
    t2, y2, ye2 : array-like
        Time vector and measurements with uncertainties for the second light curve.
    ntau : int
        Number of lag bins between tau_beg and tau_end.
    tau_beg, tau_end : float
        Start and end values of the lag grid.
    gapx, gapy : array-like or None, optional
        Gaps in the time sampling for the two light curves, given as ``(start,
        end)`` pairs. They are used to correct the effective temporal spacing when
        estimating the number of independent points.
    doplot : bool, optional
        If True, plot the results.

    Returns
    -------
    rmax, zmax : float
        ICCF rmax and zmax.
    prob_rmax : float
        Probability of ICCF rmax larger than data rmax in null hypothesis.
    """
    # calculate iccf
    tau, ccf, rmax, tau_peak, tau_cent = iccf(t1, y1, t2, y2, ntau, tau_beg, tau_end)
    zmax = np.arctanh(np.clip(rmax, -0.999, 0.999))
    print("rmax: %.2f, zmax: %.2f at tau=%.2f"%(rmax, zmax,tau_peak))

    # DRW fits 
    sample1 = drw_modeling(t1, y1, ye1, doshow=True)
    sigma1, tau1 = np.exp(np.median(sample1, axis=0))
    print("DRW sigma and tau: %.2f, %.2f"%(sigma1, tau1))

    sample2 = drw_modeling(t2, y2, ye2, doshow=True)
    sigma2, tau2 = np.exp(np.median(sample2, axis=0))
    print("DRW sigma and tau: %.2f, %.2f"%(sigma2, tau2))

    tau, nd1, ndeff1, nd2, ndeff2 =  iccf_ndeff(t1, y1, ye1, t2, y2, ye2, ntau, tau_beg, tau_end, 
                                                sigma1, tau1, sigma2, tau2, gapx=gapx, gapy=gapy)

    ndeff = 2/(1/ndeff1+1/ndeff2)
    sigma_null = 1/np.sqrt(ndeff)

    # calculate the probability of rmax in null hypothesis
    taudxy = tau1 * tau2/(tau1 + tau2)
    sig = (np.mean(1.0/ndeff))**0.5
    A = 1.0
    nm = (tau_end-tau_beg)/taudxy/2
    nm = np.max((1, nm))
    print("m of p(rmax): %.2f, taudxy: %.2f"%(nm, taudxy))
    m = np.linspace(0-sig*3, 0+sig*5, 1000)
    fm = A * nm/sig * (0.5*(1+sp.erf(m/(np.sqrt(2)*sig))))**(nm-1)*np.exp(-0.5*m**2/sig**2)/np.sqrt(2*np.pi)/sig
    idx = (m >= zmax) 
    prob_rmax = np.sum(fm[idx])/np.sum(fm)
    print("probability of rmax in null hypothesis:", prob_rmax)

    if doplot:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_axes((0.1, 0.1, 0.4, 0.8))        
        plt.plot(tau, ccf)
        ax.set_xlim(tau[0], tau[-1])
        ax.set_xlabel("Time Lag (day)")
        ax.set_ylabel(r"ICCF $(r)$")
        ax.minorticks_on()
        ax.yaxis.set_label_coords(-0.1, 0.5)

        xccf = np.linspace(tau_beg, tau_end, 200)
        yccf = np.linspace(-0.999, 0.999, 200)
        mx, my = np.meshgrid(xccf, yccf)
        sig_ccf = 1/np.interp(xccf, tau, ndeff)
        mz = np.exp(-0.5*np.arctanh(my)**2/(sig_ccf[np.newaxis, :]))
        ax.imshow(mz, aspect="auto", extent=[xccf[0], xccf[-1], yccf[0], yccf[-1]], 
                interpolation="gaussian", zorder=0, cmap="Wistia")
        ax.plot(tau, np.tanh(1.0/ndeff**0.5), ls='-', lw=1, color='gray', label=r'1$\sigma$')
        ax.plot(tau, -np.tanh(1.0/ndeff**0.5), ls='-', lw=1, color='gray')
        ax.plot(tau, np.tanh(2.0/ndeff**0.5), ls='--', lw=1, color='gray', label=r'2$\sigma$')
        ax.plot(tau, -np.tanh(2.0/ndeff**0.5), ls='--', lw=1, color='gray')
        ax.plot(tau, np.tanh(3.0/ndeff**0.5), ls=':', lw=1, color='gray', label=r'3$\sigma$')
        ax.plot(tau, -np.tanh(3.0/ndeff**0.5), ls=':', lw=1, color='gray')
        ax.legend(ncols=3, frameon=True, loc='lower center', columnspacing=1.0, 
                handlelength=1.5, handletextpad=0.4)

        ax2 = ax.twinx()
        ax2.set_ylabel(r"ICCF $(z)$")
        ax_ylim = ax.get_ylim()
        ax2.set_ylim(_ax_inverse(ax_ylim[0]), _ax_inverse(ax_ylim[1]))
        ax2.set_yscale("function", functions=(_ax_forward, _ax_inverse))

        ax = fig.add_axes((0.65, 0.1, 0.3, 0.8))
        ax.plot(m, fm)
        ax.axvline(x=zmax, ls='--', color='red')
        ylim = ax.get_ylim()
        ax.text(zmax,(ylim[0]+ylim[1])/2.0, "prob: %.2f"%prob_rmax)
        ax.set_xlabel(r"$z_{\rm max}$")
        ax.set_ylabel(r"$p(z_{\rm max})$")
        plt.show()

    return rmax, zmax, prob_rmax