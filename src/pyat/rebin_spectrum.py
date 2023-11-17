
__all__ = ["rebin_spectrum"]

import numpy as np
import matplotlib.pyplot as plt 

def rebin_spectrum(wave, prof, wave_rebin):
  """
  rebin a spectrum to an even wavelength grid
  """
  # left and right edge
  wl = wave[0] - 0.5*(wave[1]-wave[0])
  wr = wave[-1] + 0.5*(wave[-1]-wave[-2])

  # assign wave bin edge
  wave_edge = np.zeros(len(wave)+1)
  wave_edge[:-2] = wave[:-1] - 0.5*(wave[1:]-wave[0:-1])
  wave_edge[-2]  = wave[-1]  - 0.5*(wave[-1]-wave[-2])
  wave_edge[-1]  = wave[-1]  + 0.5*(wave[-1]-wave[-2])
  
  print(wave_edge)

  # assign wave rebin edge
  wave_rebin_edge = np.zeros(len(wave_rebin)+1)
  wave_rebin_edge[:-2] = wave_rebin[:-1] - 0.5*(wave_rebin[1:]-wave_rebin[0:-1])
  wave_rebin_edge[-2] = wave_rebin[-1] - 0.5*(wave_rebin[-1]-wave_rebin[-2])
  wave_rebin_edge[-1] = wave_rebin[-1] + 0.5*(wave_rebin[-1]-wave_rebin[-2])
  
  print(wave_rebin)
  print(wave_rebin_edge)

  prof_rebin = np.zeros(len(wave_rebin))
  for i in range(len(wave_rebin)):
    wbin_left, wbin_right = wave_rebin_edge[i:i+2]
    idx_left = np.searchsorted(wave_edge, wbin_left)
    idx_right = np.searchsorted(wave_edge, wbin_right)
    
    print(i, idx_left, idx_right)
    
    # if idx_left == 0:
    #   if idx_right == 0: # this rebin is smaller than wave_edge[0]
    #     prof_rebin[i] == prof[0]
    #   else:
    #     flux = prof[0] * (wave_edge[0] - wave_rebin_edge[i])

    #     for j in range(0, idx_right-1):
    #       flux += prof[j] * (wave_edge[j+1] - wave_edge[j])

    #     flux +=  (wave_rebin_edge[i+1] - wave_edge[idx_right-1]) * prof[idx_right-1]
    #     prof_rebin[i] = flux / (wave_rebin_edge[i+1]-wave_rebin_edge[i])
    
    if idx_left == idx_right:  # in the same bin of wave
      prof_rebin[i] = prof[max(0, idx_left-1)]

    else: # not in the same bin of wave
      # leftmost bin
      flux = (wave_edge[idx_left] - wave_rebin_edge[i])*prof[idx_left-1]
      # midle bins
      for j in range(idx_left, idx_right-1):
        flux += prof[j] * (wave_edge[j+1] - wave_edge[j])
      # rightmost bin
      flux += (wave_rebin_edge[i+1] - wave_edge[idx_right-1]) * prof[idx_right-1]

      prof_rebin[i] = flux / (wave_rebin_edge[i+1]-wave_rebin_edge[i])

  x = np.array(list(zip(wave_edge[:-1], wave_edge[1:]))).flatten()
  y = np.array(list(zip(prof, prof))).flatten()
  plt.plot(x, y)
  x = np.array(list(zip(wave_rebin_edge[:-1], wave_rebin_edge[1:]))).flatten()
  y = np.array(list(zip(prof_rebin, prof_rebin))).flatten()
  plt.plot(x, y)
  plt.show()


