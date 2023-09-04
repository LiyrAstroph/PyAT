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
  mean = np.squeeze(mean)
  
  # note the bias correction factor
  rms = np.sum(weight * (prof - mean)**2, axis=axis, keepdims=True)/(1.0 - np.sum(weight**2, axis=axis, keepdims=True))
  rms = np.sqrt(rms)
  rms = np.squeeze(rms)

  if return_err == True:
    mean_err = np.sqrt(np.sum(err**2*weight**2, axis=axis, keepdims=True))
    rms_err = np.sqrt(np.sum(weight**2 * (2*(prof-mean)*err)**2, axis=axis, keepdims=True))/(1.0 - np.sum(weight**2, axis=axis, keepdims=True))
    
    mean_err = np.squeeze(mean_err)
    rms_err = np.squeeze(rms_err)

  if return_err == False:
    return mean, rms
  else:
    return mean, mean_err, rms, rms_err