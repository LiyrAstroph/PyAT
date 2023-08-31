
__all__ = ["remove_outliers",]

import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage

def remove_outliers(t, y, yerr, size=1, limit=10):
  """
  Remove outliers using the median filter

  Parameters
  ----------
  t,y,yerr : 1D array like
           Input arrays.
  size : int
       Size of median filter

  limit : float
        Limit for outliers measured by the standardized deviations of points 
        from the median values

  Returns
  -------
  t, y, yerr : 1D array like
             Outliers removed arrays
  idx : 1D array like
      Indexes of outlers
  """
  ys = ndimage.median_filter(y, size=size, mode='neareast')
  std = np.abs(y-ys)/yerr 
  idx = np.where(std>=limit)[0]
  return np.delete(t, idx),np.delete(y, idx), np.delete(yerr, idx), idx
