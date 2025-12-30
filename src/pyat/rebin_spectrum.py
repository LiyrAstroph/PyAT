
#===================================================================================#
#  PyAT: Python Astronomical Tools
#  A package providing basic, common tools in astronomical analysis
#
#  Yan-Rong Li, liyropt@gmail.com
#  2023-08-31
#===================================================================================#

__all__ = ["get_bin_edge", "rebin_spectrum", "rebin_spectrum_with_error"]

import numpy as np
import matplotlib.pyplot as plt 

def get_bin_edge(wave):
  """
  get bin edges of an input wavelength grid

  Parameters
  ----------
  wave : 1D array like
    wavelength array
  
  Returns
  -------
  wave_edge : 1D array like
    wavelength bin edges
  """
  # assign wave bin edge
  wave_edge = np.zeros(len(wave)+1)
  # assign edge as middle point
  wave_edge[1:-1] = 0.5*(wave[0:-1]+wave[1:])
  # left most
  wave_edge[0] = wave[0] - 0.5*(wave[1]-wave[0])
  # right most
  wave_edge[-1]  = wave[-1]  + 0.5*(wave[-1]-wave[-2])

  return wave_edge

def rebin_spectrum(wave_rebin, wave, prof):
  """
  rebin a spectrum to an input wavelength grid

  Parameters
  ----------
  wave_rebin : 1D array like
    wavelength array rebined to.

  wave : 1D array like
    wavelength array

  prof : 1D array like
    spectrum
  
  Returns
  -------
  prof_rebin : 1D array like
    rebined spectrum
  """
  # assign wave bin edge
  wave_edge = get_bin_edge(wave)

  # assign wave rebin edge
  wave_rebin_edge = get_bin_edge(wave_rebin)

  prof_rebin = np.zeros(len(wave_rebin))
  idx_left=0
  idx_right=0
  for i in range(len(wave_rebin)):
    wbin_left, wbin_right = wave_rebin_edge[i:i+2]
    # note in numpy, slices beyond array size are fine and return an empty 
    idx_left  = np.searchsorted(wave_edge[idx_left:],  wbin_left)  + idx_left 
    idx_right = np.searchsorted(wave_edge[idx_right:], wbin_right) + idx_right
    
    # print(i, idx_left, idx_right)  
     
    if idx_left == idx_right:  # in the same bin of wave
      idx = min(max(0, idx_left-1), len(wave)-1) # make sure idx in the approprite range 
      prof_rebin[i] = prof[idx]

    else: # not in the same bin of wave
      # leftmost bin
      flux = (wave_edge[idx_left] - wave_rebin_edge[i])*prof[idx_left-1]
      # midle bins
      for j in range(idx_left, idx_right-1):
        flux += prof[j] * (wave_edge[j+1] - wave_edge[j])
      # rightmost bin
      idx = max(0, min(len(wave)-1, idx_right-1)) # make sure idx in the approprite range 
      flux += (wave_rebin_edge[i+1] - wave_edge[idx_right-1]) * prof[idx]

      prof_rebin[i] = flux / (wave_rebin_edge[i+1]-wave_rebin_edge[i])

  # x = np.array(list(zip(wave_edge[:-1], wave_edge[1:]))).flatten()
  # y = np.array(list(zip(prof, prof))).flatten()
  # plt.plot(x, y)
  # x = np.array(list(zip(wave_rebin_edge[:-1], wave_rebin_edge[1:]))).flatten()
  # y = np.array(list(zip(prof_rebin, prof_rebin))).flatten()
  # plt.plot(x, y)
  # plt.show()

  # # check flux
  # flux = 0.0
  # for i in range(len(wave)):
  #   flux += prof[i] * (wave_edge[i+1]-wave_edge[i])
  # print(flux)

  # flux = 0.0
  # for i in range(len(wave_rebin)):
  #   flux += prof_rebin[i] * (wave_rebin_edge[i+1]-wave_rebin_edge[i])
  # print(flux)

  return prof_rebin

def rebin_spectrum_with_error(wave_rebin, wave, prof, error):
  """
  rebin a spectrum to an input wavelength grid. make sure the signal to noise ratios are conserved.

  Parameters
  ----------
  wave_rebin : 1D array like
    wavelength array rebined to.

  wave : 1D array like
    wavelength array

  prof : 1D array like
    spectrum
  
  error : 1D array like
    error
   
  Returns
  -------
  prof_rebin : 1D array like
    rebined spectrum
  """
  # assign wave bin edge
  wave_edge = get_bin_edge(wave)

  # assign wave rebin edge
  wave_rebin_edge = get_bin_edge(wave_rebin)

  prof_rebin, error_rebin = np.zeros((2, len(wave_rebin)))
  idx_left=0
  idx_right=0
  for i in range(len(wave_rebin)):
    wbin_left, wbin_right = wave_rebin_edge[i:i+2]
    # note in numpy, slices beyond array size are fine and return an empty 
    idx_left  = np.searchsorted(wave_edge[idx_left:],  wbin_left)  + idx_left 
    idx_right = np.searchsorted(wave_edge[idx_right:], wbin_right) + idx_right
    
    # print(i, idx_left, idx_right)  
     
    if idx_left == idx_right:  # in the same bin of wave
      idx = min(max(0, idx_left-1), len(wave)-1) # make sure idx in the approprite range 
      prof_rebin[i]  = prof[idx]
      error_rebin[i] = error[idx]

    else: # not in the same bin of wave
      # leftmost bin
      flux = (wave_edge[idx_left] - wave_rebin_edge[i])*prof[idx_left-1]
      err = (wave_edge[idx_left] - wave_rebin_edge[i])**2 * error[idx_left-1]**2
      # midle bins
      for j in range(idx_left, idx_right-1):
        flux += prof[j] * (wave_edge[j+1] - wave_edge[j])
        err  += error[j]**2 * (wave_edge[j+1] - wave_edge[j])**2
      # rightmost bin
      idx = max(0, min(len(wave)-1, idx_right-1)) # make sure idx in the approprite range 
      flux += (wave_rebin_edge[i+1] - wave_edge[idx_right-1]) * prof[idx]
      err  += (wave_rebin_edge[i+1] - wave_edge[idx_right-1])**2 * error[idx]**2

      prof_rebin[i]  = flux / (wave_rebin_edge[i+1]-wave_rebin_edge[i])
      error_rebin[i] = np.sqrt(err) / (wave_rebin_edge[i+1]-wave_rebin_edge[i])

  # x = np.array(list(zip(wave_edge[:-1], wave_edge[1:]))).flatten()
  # y = np.array(list(zip(prof, prof))).flatten()
  # plt.plot(x, y)
  # x = np.array(list(zip(wave_rebin_edge[:-1], wave_rebin_edge[1:]))).flatten()
  # y = np.array(list(zip(prof_rebin, prof_rebin))).flatten()
  # plt.plot(x, y)
  # plt.show()

  # # check flux
  # flux = 0.0
  # for i in range(len(wave)):
  #   flux += prof[i] * (wave_edge[i+1]-wave_edge[i])
  # print(flux)

  # flux = 0.0
  # for i in range(len(wave_rebin)):
  #   flux += prof_rebin[i] * (wave_rebin_edge[i+1]-wave_rebin_edge[i])
  # print(flux)

  return prof_rebin, error_rebin