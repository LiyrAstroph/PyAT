__all__ = ["get_mean_rms",]

import numpy as np

def get_mean_rms(prof, err, axis=0, weight="uniform"):
  """
  get weighted mean and rms spectra from a set of spectra
  """
  
  if weight.lower() == "uniform":
    weight = np.ones(prof.shape)
    weight /= prof.shape[axis]

  elif weight.lower() == "sn":
    # S/N weighted 
    weight = np.abs(prof/err)
    # normalization
    weight /= np.sum(weight, axis=axis, keepdims=True) 

  elif weight.lower == "error":
    # inverse square error weighted 
    weight = 1.0/err**2
    # normalization
    weight /= np.sum(weight, axis=axis, keepdims=True) 


  mean = np.sum(prof*weight, axis=axis, keepdims=True)
  
  # no need to keepdims
  rms = np.sum(weight * (prof - mean)**2, axis=axis, keepdims=True)/(1.0 - np.sum(weight**2, axis=axis, keepdims=True))
  rms = np.sqrt(rms)
  
  return mean.flatten(), rms.flatten()