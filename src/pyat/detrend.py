#===================================================================================#
#  PyAT: Python Astronomical Tools
#  A package providing basic, common tools in astronomical analysis
#
#  Yan-Rong Li, liyropt@gmail.com
#  2023-08-31
#===================================================================================#

__all__ = ["detrend", ]

import numpy as np
from scipy.optimize import least_squares

def model0(p, x, y, yerr):
  xmed = (x[0]+x[-1])/2.0
  ymodel = p[0]+p[1]*(x-xmed)

  return (y-ymodel)**2/yerr**2

def model1(p, x, y, yerr):
  xmed = (x[0]+x[-1])/2.0
  ymodel = p[0]+p[1]*(x-xmed)+p[2]*(x-xmed)**2

  return (y-ymodel)**2/yerr**2

def detrend(x, y, yerr, return_trend=False, order=1):
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
  if order > 2: 
    raise ValueError("order should be in [1, 2]!")
  
  if order == 1:
    p=[np.mean(y), 0.0]
    res = least_squares(model0, p, args=(x, y, yerr))
  else:
    p=[np.mean(y), 0.0, 0.0]
    res = least_squares(model1, p, args=(x, y, yerr))
  
  xmed = (x[0]+x[-1])/2.0
  yd = y - res.x[1]*(x-xmed) 

  if order==2:
    yd -= res.x[2] * (x-xmed)**2

  if res.success:
    print("trend: a0 + a1 * [t - (t_beg+t_end)/2] + a2 * [t - (t_beg+t_end)/2]^2")
    for i in range(order+1):
      print("a{:d}={:.2e}".format(i, res.x[i]), end=" ")
    print()
  else:
    raise ValueError(res.message)
  
  if return_trend == False:
    return yd
  else:
    return yd, res.x

