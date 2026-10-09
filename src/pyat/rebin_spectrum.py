
#===================================================================================#
#  PyAT: Python Astronomical Tools
#  A package providing basic, common tools in astronomical analysis
#
#  Yan-Rong Li, liyropt@gmail.com
#  2023-08-31
#===================================================================================#

__all__ = ["get_bin_edge", "rebin_spectrum", "rebin_spectrum_with_error"]

import numpy as np

def get_bin_edge(wave):
  """
  get bin edges of an input wavelength grid

  Parameters
  ----------
  wave : 1D array like
    wavelength array of pixel centers, must be strictly increasing and
    contain at least two elements.

  Returns
  -------
  wave_edge : 1D array like
    wavelength bin edges, with interior edges at the midpoints between
    adjacent pixels and the two outermost edges extrapolated by half a
    pixel width.
  """
  wave = np.asarray(wave, dtype=float)

  if wave.ndim != 1 or wave.size < 2:
    raise ValueError("wave must be a 1D array with at least two elements.")
  if np.any(np.diff(wave) <= 0.0):
    raise ValueError("wave must be strictly increasing.")

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

  The spectrum is treated as piecewise constant over the input wavelength
  bins, and the rebinned values are the flux-conserving (area-weighted)
  averages over the new bins.

  Parameters
  ----------
  wave_rebin : 1D array like
    wavelength array of pixel centers rebinned to, must be strictly
    increasing and contain at least two elements.

  wave : 1D array like
    wavelength array of pixel centers, must be strictly increasing and
    contain at least two elements.

  prof : 1D array like
    spectrum, must have the same length as wave.

  Returns
  -------
  prof_rebin : 1D array like
    rebinned spectrum.

  Notes
  -----
  New bins extending beyond the wavelength range of wave are filled with
  the nearest edge pixel value (constant extrapolation); the integrated
  flux is therefore conserved only over the overlapping range.
  """
  wave = np.asarray(wave, dtype=float)
  wave_rebin = np.asarray(wave_rebin, dtype=float)
  prof = np.asarray(prof, dtype=float)

  if prof.shape != wave.shape:
    raise ValueError("prof must have the same shape as wave.")

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

    if idx_left == idx_right:  # in the same bin of wave
      idx = min(max(0, idx_left-1), len(wave)-1) # make sure idx in the appropriate range
      prof_rebin[i] = prof[idx]

    else: # not in the same bin of wave
      # leftmost bin; clamp to prof[0] when extending beyond the left edge
      idx_l = max(0, idx_left-1)
      flux = (wave_edge[idx_left] - wave_rebin_edge[i])*prof[idx_l]
      # middle bins
      for j in range(idx_left, idx_right-1):
        flux += prof[j] * (wave_edge[j+1] - wave_edge[j])
      # rightmost bin
      idx = max(0, min(len(wave)-1, idx_right-1)) # make sure idx in the appropriate range
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
  rebin a spectrum to an input wavelength grid, propagating the errors.

  The spectrum is rebinned with the same flux-conserving (area-weighted)
  averaging as rebin_spectrum. Errors of the input pixels are assumed
  independent, so the rebinned error is propagated as
  error_rebin = sqrt(sum(w_i**2 * error_i**2)) / W with w_i the
  overlapping bin widths and W the new bin width.

  Parameters
  ----------
  wave_rebin : 1D array like
    wavelength array of pixel centers rebinned to, must be strictly
    increasing and contain at least two elements.

  wave : 1D array like
    wavelength array of pixel centers, must be strictly increasing and
    contain at least two elements.

  prof : 1D array like
    spectrum, must have the same length as wave.

  error : 1D array like
    error of the spectrum, must have the same length as wave.

  Returns
  -------
  prof_rebin : 1D array like
    rebinned spectrum.
  error_rebin : 1D array like
    propagated error of the rebinned spectrum.

  Notes
  -----
  New bins extending beyond the wavelength range of wave are filled with
  the nearest edge pixel value and its error (constant extrapolation);
  the integrated flux is therefore conserved only over the overlapping
  range.
  """
  wave = np.asarray(wave, dtype=float)
  wave_rebin = np.asarray(wave_rebin, dtype=float)
  prof = np.asarray(prof, dtype=float)
  error = np.asarray(error, dtype=float)

  if prof.shape != wave.shape:
    raise ValueError("prof must have the same shape as wave.")
  if error.shape != wave.shape:
    raise ValueError("error must have the same shape as wave.")

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
      idx = min(max(0, idx_left-1), len(wave)-1) # make sure idx in the appropriate range
      prof_rebin[i]  = prof[idx]
      error_rebin[i] = error[idx]

    else: # not in the same bin of wave
      # leftmost bin; clamp to the edge pixel when extending beyond the left edge
      idx_l = max(0, idx_left-1)
      w_left = wave_edge[idx_left] - wave_rebin_edge[i]
      flux = w_left*prof[idx_l]
      err = w_left**2 * error[idx_l]**2
      # middle bins
      for j in range(idx_left, idx_right-1):
        w_mid = wave_edge[j+1] - wave_edge[j]
        flux += prof[j] * w_mid
        err  += error[j]**2 * w_mid**2
      # rightmost bin
      idx = max(0, min(len(wave)-1, idx_right-1)) # make sure idx in the appropriate range
      w_right = wave_rebin_edge[i+1] - wave_edge[idx_right-1]
      flux += w_right * prof[idx]
      err  += w_right**2 * error[idx]**2

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