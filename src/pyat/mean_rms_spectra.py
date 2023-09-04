__all__ = ["get_mean_rms",]

import numpy as np

def get_mean_rms(prof, err, axis=0, weight="uniform", return_err=False):
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
  
  # note the bias correction factor
  rms = np.sum(weight * (prof - mean)**2, axis=axis, keepdims=True)/(1.0 - np.sum(weight**2, axis=axis, keepdims=True))
  rms = np.sqrt(rms)

  if return_err == True:
    mean_err = np.sqrt(np.sum(err**2*weight**2, axis=axis, keepdims=True))
    rms_err = np.sqrt(np.sum(weight**2 * (2*(prof-mean)*err)**2, axis=axis, keepdims=True))/(1.0 - np.sum(weight**2, axis=axis, keepdims=True))
  
  if return_err == False:
    return mean.flatten(), rms.flatten()
  else:
    return mean.flatten(), mean_err.flatten(), rms.flatten(), rms_err.flatten()