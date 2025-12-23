
__all__ = ["drw_recon", "drw_modeling", "genlc_psd_pow", "genlc_psd_drw"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import celerite
from celerite import terms
from celerite.modeling import Model
from numpy import fft 

import emcee
import corner

def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def log_probability(params, y, gp):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    return gp.log_likelihood(y) + lp

def drw_recon(t, y, yerr):
    """
    reconstruct a light curve using the DRW model

    Parameters
    ----------
    t,y,yerr : 1D array like
            Input arrays.

    Returns
    -------
    t, y, yerr : 1D array like
                Outliers removed arrays
    """
    t0 = t[0]
    tnew = t - t0

    # Damped random walk model
    log_a = 1.5
    log_c = -np.log((tnew[-1]-tnew[0])/2.0)
    bounds2 = dict(log_a=(-10, 10), log_c=(-np.log(tnew[-1]-tnew[0]), 0))
    kernel2 = terms.RealTerm(log_a=log_a, log_c=log_c, bounds=bounds2)
    kernel = kernel2


    # build celerite model
    gp = celerite.GP(kernel, mean=np.mean(y))
    gp.compute(t, yerr)

    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()

    r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gp))
    initial_params = gp.get_parameter_vector()
    print(initial_params)
    print(np.exp(0.5*initial_params[0]), np.exp(-initial_params[1]))

    # Time grid for reconstruction
    tspan = t[-1]-t[0]
    x = np.linspace(t[0]-0.05*tspan, t[-1]+0.05*tspan, 500)

    pred_mean, pred_var = gp.predict(y, x, return_var=True)
    pred_std = np.sqrt(pred_var)

    return x, pred_mean, pred_std

def drw_modeling(t, y, yerr, doshow=False):
    """
    Determine DRW parameters for a given light curve
    
    :param t: time
    :param y: flux
    :param yerr: error
    :param doshow: whether show figures
    """

    # first normalize the light curve 
    scale = -np.ceil(np.log10(np.median(y)))
    y_new = y * 10.0**(scale) 
    yerr_new = yerr * 10.0**(scale) 

    t_new = t - t[0] 

    # Damped random walk model
    log_a = 1.5
    log_c = -4.5
    bounds2 = dict(log_a=(-10, 10), log_c=(-np.log(t_new[-1]-t_new[0]), 0))
    kernel2 = terms.RealTerm(log_a=log_a, log_c=log_c, bounds=bounds2)
    kernel = kernel2


    # build celerite model
    gp = celerite.GP(kernel, mean=np.mean(y_new))
    gp.compute(t_new, yerr_new)

    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()

    r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(y_new, gp))
    initial_params = gp.get_parameter_vector()
    print(initial_params)
    print(np.exp(0.5*initial_params[0]), np.exp(-initial_params[1]))

    # Time grid for reconstruction
    x = np.linspace(t_new[0]-10.0, t_new[-1]+10.0, np.max((1000, int(t_new[-1]+20.0-t_new[0]))))

    pred_mean, pred_var = gp.predict(y_new, x, return_var=True)
    pred_std = np.sqrt(pred_var)

    color = "#ff7f0e"
    plt.errorbar(t_new, y_new, yerr=yerr_new, fmt=".k", capsize=0)
    plt.plot(x, pred_mean, color=color)
    plt.fill_between(x, pred_mean+pred_std, pred_mean-pred_std, color=color, alpha=0.3,
                    edgecolor="none")
    plt.xlabel("Time")
    plt.ylabel("Flux ($10^{%d}$)"%(scale))
    if doshow:
        plt.show()
    else:
        plt.close()


    # MCMC sampling
    initial = initial_params
    ndim, nwalkers = len(initial), 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(y_new, gp))

    print("Running burn-in...")
    p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
    p0, lp, _ = sampler.run_mcmc(p0, 500)

    print("Running production...")
    sampler.reset()
    sampler.run_mcmc(p0, 2000)

    sample = sampler.flatchain
    sample[:, 0] = sample[:, 0]*0.5 - np.log(10.0**scale)
    sample[:, 1] = -sample[:, 1]

    if doshow:
        corner.corner(sample, labels=["ln(sigma)", "ln(tau)"])
        plt.show()
    
    return sample

#=======================================================
# DRW PSD function
#
#=======================================================
def psd_drw(fk, arg, freq_limit):
    sigma, tau, cnoise = arg[:3]

    nud = 1.0/(2.0*np.pi*tau)
    psd = np.zeros(fk.shape)

    psd[:] = 2.0*sigma**2*tau/(1.0 + (fk/nud)**2) + cnoise

    return psd

#=======================================================
# square root of DRW PSD function
#
#=======================================================
def psd_drw_sqrt(fk, arg, freq_limit):
    sigma, tau, cnoise = arg[:3]
    nud = 1.0/(2.0*np.pi*tau)
    psd = np.zeros(fk.shape)

    psd[:] = 2.0*sigma**2*tau/(1.0 + (fk/nud)**2) + cnoise

    return np.sqrt(psd)


#=======================================================
# single power law PSD function
#
#=======================================================
def psd_power_law(fk, arg, freq_limit):
    A, alpha, cnoise =arg[:3]

    psd = np.zeros(fk.shape)

    idx = np.where(fk >= freq_limit)
    psd[idx[0]] = A * fk[idx[0]]**(-alpha) + cnoise
    idx = np.where(fk < freq_limit)
    psd[idx[0]] = A * freq_limit**(-alpha) + cnoise

    return psd

#=======================================================
# square root of single power law PSD function
#
#=======================================================
def psd_power_law_sqrt(fk, arg, freq_limit):
    A, alpha, cnoise =arg[:3]

    psd = np.zeros(fk.shape)

    idx = np.where(fk >= freq_limit)
    psd[idx[0]] = A * fk[idx[0]]**(-alpha) + cnoise
    idx = np.where(fk < freq_limit)
    psd[idx[0]] = A * freq_limit**(-alpha) + cnoise

    return np.sqrt(psd)

#=======================================================  
# generate time series with given power-law PSD
#
# note that integrating PSD over (-infty, +infty) equals 
# to variation of light curves
#=======================================================
def genlc_psd_pow(model, nd, DT, freq_limit):  
    W = 4
    V = 4
    nd_sim = int(W*V*nd)
    arg = model
    fft_work = np.zeros(nd_sim//2+1, dtype=complex)

    fft_work[0] = np.random.randn()+1j*0.0

    freq = 1.0/(nd_sim * DT/W) * np.linspace(0.0, nd_sim//2, nd_sim//2+1)
    fft_work[1:nd_sim//2] = psd_power_law_sqrt(freq[1:nd_sim//2], arg, freq_limit)/np.sqrt(2.0) \
                        * ( np.random.randn(int(nd_sim//2-1)) + 1j*np.random.randn(int(nd_sim//2-1)) )
    fft_work[nd_sim//2] = psd_power_law_sqrt(freq[nd_sim//2:], arg, freq_limit) * (np.random.randn() + 1j*0.0)

    fs = fft.irfft(fft_work) * nd_sim # note the factor 1/n in numpy ifft,

    norm = 1.0/np.sqrt(nd_sim) * np.sqrt(nd_sim/(2.0*nd_sim * DT))

    ts = DT/W * ( np.linspace(0, nd_sim-1, nd_sim) - nd_sim/2.0)
    fs = fs*norm

    return ts[::W], fs[::W]
  
#=======================================================  
# generate time series with given drw PSD
# 
# note that integrating PSD over (-infty, +infty) equals 
# to variation of light curves
#=======================================================
def genlc_psd_drw(model, nd, DT, freq_limit):  
    W = 4
    V = 4
    nd_sim = int(W*V*nd)
    arg = model
    fft_work = np.zeros(nd_sim//2+1, dtype=complex)

    fft_work[0] = np.random.randn()+1j*0.0

    freq = 1.0/(nd_sim * DT/W) * np.linspace(0.0, nd_sim//2, nd_sim//2+1)
    fft_work[1:nd_sim//2] = 2.0**0.5*psd_drw_sqrt(freq[1:nd_sim//2], arg, freq_limit)/np.sqrt(2.0) \
                        * ( np.random.randn(int(nd_sim//2-1)) + 1j*np.random.randn(int(nd_sim//2-1)) )
    fft_work[nd_sim//2] = 2.0**0.5*psd_drw_sqrt(freq[nd_sim//2:], arg, freq_limit) * (np.random.randn() + 1j*0.0)

    fs = fft.irfft(fft_work) * nd_sim # note the factor 1/n in numpy ifft,

    norm = 1.0/np.sqrt(nd_sim) * np.sqrt(nd_sim/(2.0*nd_sim * DT))

    ts = DT/W * ( np.linspace(0, nd_sim-1, nd_sim) - nd_sim/2.0)
    fs = fs*norm

    return ts[::W], fs[::W]  