#===================================================================================#
#  PyAT: Python Astronomical Tools
#  A package providing basic, common tools in astronomical analysis
#
#  Yan-Rong Li, liyropt@gmail.com
#  2023-08-31
#===================================================================================#

__all__ = ["estimate_syserr", "estimate_syserr_median_filter"]

import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage

def estimate_syserr_median_filter(y, yerr, size=5, return_yfilter=False):
  """
  estimate systematic errors using the median filter.

  Parameters
  ----------
  y, yerr : 1D array like
    Input arrays

  size : int
    Size of median filter 

  return_yfilter : boolen, optional
    Whether return the filtered result of y.

  Returns
  ------- 
  syserr : float
    The estimated systematic error

  y_filter: 1D array like, optinal
    The filtered result of y            
  """
  
  y_filter = ndimage.median_filter(y, size=(size//2)*2+1, mode='nearest')

  dev = np.mean((y-y_filter)**2)
  err = np.mean(yerr**2)

  if dev < err:
    syserr = 0.0
  else:
    syserr =  np.sqrt(dev - err)
  
  if return_yfilter == False:
    return syserr
  else:
    return syserr, y_filter

def estimate_syserr(t, y, yerr, size=5, return_yfilter=False):
  """
  estimate systematic errors.

  Parameters
  ----------
  t, y, yerr : 1D array like
    Input arrays, time, flux, and error

  size : int
    Size of median filter 

  return_yfilter : boolen, optional
    Whether return the filtered result of y.

  Returns
  ------- 
  syserr : float
    The estimated systematic error

  y_filter: 1D array like, optinal
    The filtered result of y      
  """
  dt = t[1:]-t[:-1]
  if np.min(dt) < 0.0:
    raise ValueError("time is not monotonously increasing!")
  
  if len(t) != len(y) or len(t) != len(yerr):
    raise ValueError("dimensions of t, y, and yerr are not same!")
  
  if np.abs(np.min(dt)/np.max(dt) - 1.0) < 0.1:  # nearly evenly spaced

    syserr, y_filter = estimate_syserr_median_filter(y, yerr, size=size, return_yfilter=True)

  else: # unevenly spaced

    dtwin = (np.median(dt) * size)/2 # half window

    y_filter = np.zeros(len(y))
    num = np.zeros(len(y), dtype=int)
    for i in range(len(t)):
      idx = np.where((t>=t[i]-dtwin) & (t<=t[i]+dtwin))[0]
      y_filter[i] = np.median(y[idx])
      num[i] = len(idx)
    
    # exclude points without neighbours in the median filter window
    idx = np.where(num>1)[0]
    dev = np.mean((y[idx]-y_filter[idx])**2)
    err = np.mean(yerr[idx]**2)

    if dev < err:
      syserr = 0.0
    else:
      syserr = np.sqrt(dev - err)
  
  if return_yfilter == False:
    return syserr
  else:
    return syserr, y_filter