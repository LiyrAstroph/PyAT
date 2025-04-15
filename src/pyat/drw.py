
__all__ = ["drw_recon",]

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import celerite
from celerite import terms
from celerite.modeling import Model

import emcee
import corner

def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

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
    