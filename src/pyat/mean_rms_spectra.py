__all__ = ["get_mean_rms", "get_line_widths"]

import numpy as np
import copy
import matplotlib.pyplot as plt 

def get_mean_rms(prof, err, axis=0, weight="uniform", return_err=False):
  """
  get weighted mean and rms spectra from a set of spectra.

  Parameters
  ----------
  prof : array like 
    The input line profile
  
  err : array like 
    The input errors
  
  axis : int 
    The axis along which to calculate mean and rms
  
  weight : string {"uniform", "sn", "error"}, optional
    The weight.
  
  return_err : boolen, optinal
    Whether return errors of mean and rms spectra.
  
  Returns
  -------
  mean, (mean_err), rms, (rms_err) : array like
    The mean, rms spectra and their errors if return_err is True.
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

def get_line_widths(wave, prof, line_win=None, flag_con_sub=False, con_sub_win=None, doplot=False):
  """
  calculate line widths from a given profile

  Parameters
  ----------
  wave : 1D array like
    wavelength
  
  prof : 1D array like 
    line profile
  
  line_win : {w1, w2}, float
    line window
  
  flag_con_sub : boolen, optional
    whether subtracting the underlying continuum
  
  con_sun_win : {w1, w2}, float, optional 
    the windows to subtacting the underlying continuum

  Returns
  -------
  fwhm, sigma: float
    FWHM and sigma.

  """
  prof_sub = copy.copy(prof)
  
  if flag_con_sub == True:
    contl1, contl2, contr1, contr2 = np.array(con_sub_win)
    wave_left = 0.5*(contl1+contl2)
    wave_right = 0.5*(contr1+contr2)
    idx_left = np.where((wave>=contl1)&(wave<=contl2))
    idx_right = np.where((wave>=contr1)&(wave<=contr2))
    con_left = np.mean(prof[idx_left])
    con_right = np.mean(prof[idx_right])
    con_sub = (con_right - con_left)/(wave_right - wave_left) *(wave - wave_left) + con_left
    prof_sub[:] -= con_sub
  
  # extract the line region
  idx = np.where((wave>=line_win[0]) & (wave<=line_win[1]))[0]
  wave_win = wave[idx]
  prof_win = prof_sub[idx]
  
  # get the peak wavelength and flux
  imax = np.argmax(prof_win)
  wmax = wave_win[imax]
  fmax = prof_win[imax]
  
  wl = np.interp(fmax*0.5, prof_win[:imax], wave_win[:imax])
  wr = np.interp(fmax*0.5, prof_win[-1:imax:-1], wave_win[-1:imax:-1])
  fwhm = wr-wl
  
  # get sigma 
  wmean = np.sum(prof_win*wave_win)/np.sum(prof_win)
  sigma = np.sqrt(np.sum(prof_win * (wave_win - wmean)**2)/np.sum(prof_win))
  
  if doplot:
    plt.ion()

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(wave, prof_sub)
    if flag_con_sub:
      ax.plot(wave, prof)
      ylim = ax.get_ylim()
      ax.fill_between(x=con_sub_win[0:2], y1=[ylim[0], ylim[0]], y2=[ylim[1], ylim[1]], color='gainsboro', zorder=0)
      ax.fill_between(x=con_sub_win[2:4], y1=[ylim[0], ylim[0]], y2=[ylim[1], ylim[1]], color='gainsboro', zorder=0)
      ax.fill_between(x=line_win, y1=[ylim[0], ylim[0]], y2=[ylim[1], ylim[1]], color='gainsboro', zorder=0)
      ax.set_ylim(ylim[0], ylim[1])

    ax.axhline(y=fmax, ls='--', lw=1, color='grey')
    ax.axhline(y=fmax*0.5, ls='--',lw=1, color='grey')
    ax.axvline(x=wr,ls='--', lw=1, color='grey')
    ax.axvline(x=wl,ls='--', lw=1, color='grey')
    ax.axhline(y=0,ls='--', lw=1, color='grey')
    
    plt.ioff()

  return fwhm, sigma