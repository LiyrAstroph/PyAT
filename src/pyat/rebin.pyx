#!/usr/bin/python
#cython: initializedcheck=False, boundscheck=False, wraparound=False, cdivision=True, profile=False

__all__ = ["rebin", "rebin_sig"]

cimport cython

import numpy as np
cimport numpy as np
from libc.math cimport sqrt


# get the mean stand deviation
cpdef double get_sig_mean(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y, double tb):

  # first check if x is monotonously increasing
  dx = x[1:] - x[:-1]
  if np.count_nonzero(dx < 0) > 0:
    raise ValueError("time is not monotonously increasing.")

  cdef double sig_mean, mean, sig
  cdef unsigned int nt, jc, i, j, j1, j2
  
  sig_mean = 0.0
  nt = 0
  for i in range(len(x)): 
    jc =  0
    mean = 0
    for j in range(len(x)):
      if (x[j]>x[i]-tb/2.0) and (x[j] < x[i] + tb/2.0):
        jc += 1
        mean += y[j]
           
    if jc > 1:
      mean /= jc
      sig = 0.0
      for j in range(len(x)):
        if (x[j]>x[i]-tb/2.0) and (x[j] < x[i] + tb/2.0):
          sig += (y[j] - mean) * (y[j] - mean)
      
      sig_mean += sig/(jc - 1)
      nt += 1
      
  sig_mean = sqrt(sig_mean/nt)
  
  return sig_mean

# rebin the lightcurve
cpdef rebin(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] ye, double tb):
   
  # first check if x is monotonously increasing
  dx = x[1:] - x[:-1]
  if np.count_nonzero(dx < 0) > 0:
    raise ValueError("time is not monotonously increasing.")

  cdef unsigned i, j, nc, jp, ic
  cdef double sig_mean, tmean, fmean, error, sig, ye2, norm
  
  i = 0
  ic = 0
  xc = np.zeros(len(x), dtype=np.double)
  yc = np.zeros(len(x), dtype=np.double)
  yerr = np.zeros(len(x), dtype=np.double)
  
  while i < len(x):
    tmean = 0.0
    fmean = 0.0
    norm = 0.0
    nc = 0
    for j in range(i, len(x)):
      if x[j] < x[i] + tb:
        ye2 = ye[j]*ye[j]
        tmean += x[j]/(ye2)
        fmean += y[j]/(ye2)
        norm += 1.0/ye2
        nc += 1
      
    jp = nc + i - 1
           
    tmean /= norm
    fmean /= norm
    error = sqrt(1.0/norm)
      
    sig = error
    
    xc[ic] = tmean
    yc[ic] = fmean
    yerr[ic] = sig
    
    i = jp  + 1
    ic += 1
     
  return xc[:ic], yc[:ic], yerr[:ic]

# rebin the lightcurve
cpdef void rebin_sig(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] ye, double tb, 
                     np.ndarray[double, ndim=1] xc, np.ndarray[double, ndim=1] yc, np.ndarray[double, ndim=1] yerr):
  
  # first check if x is monotonously increasing
  dx = x[1:] - x[:-1]
  if np.count_nonzero(dx < 0) > 0:
    raise ValueError("time is not monotonously increasing.")
    
  cdef unsigned int i, j, nc, jp, ic
  cdef double sig_mean, tmean, fmean, error, sig, ye2, norm, mean

  sig_mean = get_sig_mean(x,y,tb)
  #print("sig_mean:", sig_mean)
  
  i = 0
  ic = 0
  
  while i < len(x):
    tmean = 0.0
    fmean = 0.0
    norm = 0.0
    mean = 0.0
    nc = 0
    for j in range(i, len(x)):
      if x[j] < x[i] + tb:
        ye2 = ye[j]*ye[j]
        tmean += x[j]/(ye2)
        fmean += y[j]/(ye2)
        norm += 1.0/ye2
        mean += y[j]
        nc += 1
      
    jp = nc + i - 1
           
    tmean /= norm
    fmean /= norm
    error = sqrt(1.0/norm)
    
    if(nc>1):
      mean /= nc
      sig = 0.0
      for j in range(i, len(x)):
        if x[j] < x[i] + tb:
          sig += (y[j] - mean)*(y[j] - mean)
          
      sig = sqrt(sig/(nc-1))
   
    else:
      sig = error
      
    sig = np.max((sig, error, sig_mean))
    
    xc[ic] = tmean
    yc[ic] = fmean
    yerr[ic] = sig
    
    i = jp  + 1
    ic += 1

  return

cpdef get_size_sig(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] ye, double tb):
  
  # first check if x is monotonously increasing
  dx = x[1:] - x[:-1]
  if np.count_nonzero(dx < 0) > 0:
    raise ValueError("time is not monotonously increasing.")
    
  cdef unsigned int i, j, nc, jp
  
  i = 0
  ic = 0
  while i < len(x):
    nc = 0
    for j in range(i, len(x)):
      if x[j] < x[i] + tb:
        nc += 1
      
    jp = nc + i - 1
       
    i = jp  + 1
    ic += 1
    
  return ic
  
