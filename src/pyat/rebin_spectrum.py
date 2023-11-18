
__all__ = ["rebin_spectrum"]

import numpy as np
import matplotlib.pyplot as plt 

def rebin_spectrum(wave, prof, wave_rebin):
  """
  rebin a spectrum to an even wavelength grid
  """
  # assign wave bin edge
  wave_edge = np.zeros(len(wave)+1)
  # assign edge as middle point
  wave_edge[1:-1] = 0.5*(wave[0:-1]+wave[1:])
  # left most
  wave_edge[0] = wave[0] - 0.5*(wave[1]-wave[0])
  # right most
  wave_edge[-1]  = wave[-1]  + 0.5*(wave[-1]-wave[-2])

  # assign wave rebin edge
  wave_rebin_edge = np.zeros(len(wave_rebin)+1)
  # assign edge as middle point
  wave_rebin_edge[1:-1] = 0.5*(wave_rebin[0:-1]+wave_rebin[1:])
  # left most
  wave_rebin_edge[0] = wave_rebin[0] - 0.5*(wave_rebin[1]-wave_rebin[0])
  # right most
  wave_rebin_edge[-1]  = wave_rebin[-1]  + 0.5*(wave_rebin[-1]-wave_rebin[-2])

  prof_rebin = np.zeros(len(wave_rebin))
  idx_left=0
  idx_right=0
  for i in range(len(wave_rebin)):
    wbin_left, wbin_right = wave_rebin_edge[i:i+2]
    idx_left = np.searchsorted(wave_edge[idx_left:], wbin_left) + idx_left
    idx_right = np.searchsorted(wave_edge[idx_right:], wbin_right) + idx_right
    
    print(i, idx_left, idx_right)  
     
    if idx_left == idx_right:  # in the same bin of wave
      idx = min(max(0, idx_left-1), len(wave)-1) # make sure idx in the approporite range 
      prof_rebin[i] = prof[idx]

    else: # not in the same bin of wave
      # leftmost bin
      flux = (wave_edge[idx_left] - wave_rebin_edge[i])*prof[idx_left-1]
      # midle bins
      for j in range(idx_left, idx_right-1):
        flux += prof[j] * (wave_edge[j+1] - wave_edge[j])
      # rightmost bin
      idx = max(0, min(len(wave)-1, idx_right-1)) # make sure idx in the approporite range 
      flux += (wave_rebin_edge[i+1] - wave_edge[idx_right-1]) * prof[idx]

      prof_rebin[i] = flux / (wave_rebin_edge[i+1]-wave_rebin_edge[i])

  x = np.array(list(zip(wave_edge[:-1], wave_edge[1:]))).flatten()
  y = np.array(list(zip(prof, prof))).flatten()
  plt.plot(x, y)
  x = np.array(list(zip(wave_rebin_edge[:-1], wave_rebin_edge[1:]))).flatten()
  y = np.array(list(zip(prof_rebin, prof_rebin))).flatten()
  plt.plot(x, y)
  plt.show()

  # check flux
  flux = 0.0
  for i in range(len(wave)):
    flux += prof[i] * (wave_edge[i+1]-wave_edge[i])
  print(flux)

  flux = 0.0
  for i in range(len(wave_rebin)):
    flux += prof_rebin[i] * (wave_rebin_edge[i+1]-wave_rebin_edge[i])
  print(flux)

  return prof_rebin


