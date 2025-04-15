__all__ = ["detrend", ]

import numpy as np
from scipy.optimize import least_squares

def model(p, x, y, yerr):
  xmed = (x[0]+x[-1])/2.0
  ymodel = p[0]+p[1]*(x-xmed)

  return (y-ymodel)**2/yerr**2

def detrend(x, y, yerr, return_trend=False):
  """
  detrend a light curve using a linear line.

  Parameters
  ----------
  t, y, yerr : 1D array like
        Input arrays.

  Returns
  -------
  y : 1D array like
        Detrended arrays
  """
  
  p=[0.0, 0.0]
  res = least_squares(model, p, args=(x, y, yerr))
  
  xmed = (x[0]+x[-1])/2.0
  yd = y - res.x[1]*(x-xmed) 

  if res.success:
    print("trend: a + b * [t - (t_beg+t_end)/2]\na={:.2e}, b={:.2e}".format(res.x[0], res.x[1]))
  else:
    raise ValueError(res.message)
  
  if return_trend == False:
    return yd
  else:
    return yd, res.x

