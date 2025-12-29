#!/usr/bin/python
#cython: initializedcheck=False, boundscheck=False, wraparound=False, cdivision=True, profile=False

__all__ = ["iccf", "iccf_mc", "iccf_oneway", "iccf_mc_oneway"]

cimport cython

import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

np.import_array()

cpdef iccf(
  np.ndarray[double, ndim=1] t1, 
  np.ndarray[double, ndim=1] f1, 
  np.ndarray[double, ndim=1] t2, 
  np.ndarray[double, ndim=1] f2, 
  int ntau, 
  double tau_beg, 
  double tau_end, 
  double threshold=0.8, 
  str mode="multiple", 
  bint ignore_warning=False):
  """
  python interface for iccf 
  """
  
  if mode not in ["multiple", "single"]:
    raise ValueError("mode = %s is not recognized! use 'multiple' or 'single'!"%mode)

  cdef char mode_cython 
  if mode == "multiple":
    mode_cython = "m"
  else:
    mode_cython = "s"
  
  cdef np.ndarray[double, ndim=1] tau = np.linspace(tau_beg, tau_end, ntau)
  cdef np.ndarray[double, ndim=1] ccf = np.zeros(ntau)
  cdef double rmax, tau_peak, tau_cent
  
  cdef double *t1_cython = <double *>PyMem_Malloc(t1.shape[0]*sizeof(double))
  cdef double *f1_cython = <double *>PyMem_Malloc(t1.shape[0]*sizeof(double))
  cdef double *t2_cython = <double *>PyMem_Malloc(t2.shape[0]*sizeof(double))
  cdef double *f2_cython = <double *>PyMem_Malloc(t2.shape[0]*sizeof(double))

  cdef double *tau_cython = <double *>PyMem_Malloc(ntau*sizeof(double))
  cdef double *ccf_cython = <double *>PyMem_Malloc(ntau*sizeof(double))

  cdef int i

  if t2[0] > t1[t1.shape[0]-1] or t1[0] > t2[t2.shape[0]-1]:
    print("no overlap, set ccf to zero.")
    for i in range(ntau):
      tau[i] = tau_beg + (tau_end-tau_beg)/(ntau-1) * i 
      ccf[i] = 0.0
    return tau, ccf, -10.0, 0.0, 0.0

  for i in range(t1.shape[0]):
    t1_cython[i] = t1[i]
    f1_cython[i] = f1[i]
  
  for i in range(t2.shape[0]):
    t2_cython[i] = t2[i]
    f2_cython[i] = f2[i]

  # call c function 
  ciccf(t1_cython, f1_cython, t1.shape[0], t2_cython, f2_cython, t2.shape[0],
        ntau, tau_beg, tau_end, threshold, mode_cython, ignore_warning,
        tau_cython, ccf_cython, &rmax, &tau_peak, &tau_cent)
  
  for i in range(ntau):
    tau[i] = tau_cython[i]
    ccf[i] = ccf_cython[i]

  PyMem_Free(t1_cython)
  PyMem_Free(f1_cython)
  PyMem_Free(t2_cython)
  PyMem_Free(f2_cython)

  PyMem_Free(tau_cython)
  PyMem_Free(ccf_cython)

  return tau, ccf, rmax, tau_peak, tau_cent

cpdef iccf_mc(
  np.ndarray[double, ndim=1] t1, 
  np.ndarray[double, ndim=1] f1, 
  np.ndarray[double, ndim=1] e1,
  np.ndarray[double, ndim=1] t2, 
  np.ndarray[double, ndim=1] f2,
  np.ndarray[double, ndim=1] e2, 
  int ntau, 
  double tau_beg, 
  double tau_end, 
  int nsim=1000,
  double threshold=0.8, 
  str mode="multiple", 
  bint ignore_warning=False):
  """
  ICCF monte carlo
  """
  
  if mode not in ["multiple", "single"]:
    raise ValueError("mode = %s is not recognized! use 'multiple' or 'single'!"%mode)

  cdef char mode_cython 
  if mode == "multiple":
    mode_cython = "m"
  else:
    mode_cython = "s"
  
  cdef np.ndarray[double, ndim=1] ccf_peak_mc = np.zeros(nsim)
  cdef np.ndarray[double, ndim=1] tau_peak_mc = np.zeros(nsim)
  cdef np.ndarray[double, ndim=1] tau_cent_mc = np.zeros(nsim)

  cdef double *t1_cython = <double *>PyMem_Malloc(t1.shape[0]*sizeof(double))
  cdef double *f1_cython = <double *>PyMem_Malloc(t1.shape[0]*sizeof(double))
  cdef double *e1_cython = <double *>PyMem_Malloc(t1.shape[0]*sizeof(double))
  cdef double *t2_cython = <double *>PyMem_Malloc(t2.shape[0]*sizeof(double))
  cdef double *f2_cython = <double *>PyMem_Malloc(t2.shape[0]*sizeof(double))
  cdef double *e2_cython = <double *>PyMem_Malloc(t2.shape[0]*sizeof(double))

  cdef double *ccf_peak_mc_cython = <double *>PyMem_Malloc(nsim*sizeof(double))
  cdef double *tau_peak_mc_cython = <double *>PyMem_Malloc(nsim*sizeof(double))
  cdef double *tau_cent_mc_cython = <double *>PyMem_Malloc(nsim*sizeof(double))
  
  for i in range(t1.shape[0]):
    t1_cython[i] = t1[i]
    f1_cython[i] = f1[i]
    e1_cython[i] = e1[i]
  
  for i in range(t2.shape[0]):
    t2_cython[i] = t2[i]
    f2_cython[i] = f2[i]
    e2_cython[i] = e2[i]

  # call c function  
  ciccf_mc(t1_cython, f1_cython, e1_cython, t1.shape[0],
           t2_cython, f2_cython, e2_cython, t2.shape[0],
           ntau, tau_beg, tau_end, nsim, 
           threshold, mode_cython, ignore_warning,
           ccf_peak_mc_cython, tau_peak_mc_cython, tau_cent_mc_cython)
  
  for i in range(nsim):
    ccf_peak_mc[i] = ccf_peak_mc_cython[i]
    tau_peak_mc[i] = tau_peak_mc_cython[i]
    tau_cent_mc[i] = tau_cent_mc_cython[i]

  PyMem_Free(t1_cython)
  PyMem_Free(f1_cython)
  PyMem_Free(e1_cython)
  PyMem_Free(t2_cython)
  PyMem_Free(f2_cython)
  PyMem_Free(e2_cython)

  PyMem_Free(ccf_peak_mc_cython)
  PyMem_Free(tau_peak_mc_cython)
  PyMem_Free(tau_cent_mc_cython)
  return ccf_peak_mc, tau_peak_mc, tau_cent_mc

cpdef iccf_oneway(
  np.ndarray[double, ndim=1] t1, 
  np.ndarray[double, ndim=1] f1, 
  np.ndarray[double, ndim=1] t2, 
  np.ndarray[double, ndim=1] f2, 
  int ntau, 
  double tau_beg, 
  double tau_end, 
  double threshold=0.8, 
  str mode="multiple", 
  bint ignore_warning=False):

  """
  one-way ICCF
  """

  if mode not in ["multiple", "single"]:
    raise ValueError("mode = %s is not recognized! use 'multiple' or 'single'!"%mode)

  cdef char mode_cython 
  if mode == "multiple":
    mode_cython = "m"
  else:
    mode_cython = "s"
  
  cdef np.ndarray[double, ndim=1] tau = np.linspace(tau_beg, tau_end, ntau)
  cdef np.ndarray[double, ndim=1] ccf = np.zeros(ntau)
  cdef double rmax, tau_peak, tau_cent
  
  cdef double *t1_cython = <double *>PyMem_Malloc(t1.shape[0]*sizeof(double))
  cdef double *f1_cython = <double *>PyMem_Malloc(t1.shape[0]*sizeof(double))
  cdef double *t2_cython = <double *>PyMem_Malloc(t2.shape[0]*sizeof(double))
  cdef double *f2_cython = <double *>PyMem_Malloc(t2.shape[0]*sizeof(double))

  cdef double *tau_cython = <double *>PyMem_Malloc(ntau*sizeof(double))
  cdef double *ccf_cython = <double *>PyMem_Malloc(ntau*sizeof(double))

  cdef int i

  if t2[0] > t1[t1.shape[0]-1] or t1[0] > t2[t2.shape[0]-1]:
    print("no overlap, set ccf to zero.")
    for i in range(ntau):
      tau[i] = tau_beg + (tau_end-tau_beg)/(ntau-1) * i 
      ccf[i] = 0.0
    return tau, ccf, -10.0, 0.0, 0.0

  for i in range(t1.shape[0]):
    t1_cython[i] = t1[i]
    f1_cython[i] = f1[i]
  
  for i in range(t2.shape[0]):
    t2_cython[i] = t2[i]
    f2_cython[i] = f2[i]

  # call c function 
  ciccf_oneway(t1_cython, f1_cython, t1.shape[0], t2_cython, f2_cython, t2.shape[0],
        ntau, tau_beg, tau_end, threshold, mode_cython, ignore_warning,
        tau_cython, ccf_cython, &rmax, &tau_peak, &tau_cent)
  
  for i in range(ntau):
    tau[i] = tau_cython[i]
    ccf[i] = ccf_cython[i]

  PyMem_Free(t1_cython)
  PyMem_Free(f1_cython)
  PyMem_Free(t2_cython)
  PyMem_Free(f2_cython)

  PyMem_Free(tau_cython)
  PyMem_Free(ccf_cython)

  return tau, ccf, rmax, tau_peak, tau_cent

cpdef iccf_mc_oneway(
  np.ndarray[double, ndim=1] t1, 
  np.ndarray[double, ndim=1] f1, 
  np.ndarray[double, ndim=1] e1,
  np.ndarray[double, ndim=1] t2, 
  np.ndarray[double, ndim=1] f2,
  np.ndarray[double, ndim=1] e2, 
  int ntau, 
  double tau_beg, 
  double tau_end, 
  int nsim=1000,
  double threshold=0.8, 
  str mode="multiple", 
  bint ignore_warning=False):
  """
  one-way ICCF monte carlo
  """
  
  if mode not in ["multiple", "single"]:
    raise ValueError("mode = %s is not recognized! use 'multiple' or 'single'!"%mode)

  cdef char mode_cython 
  if mode == "multiple":
    mode_cython = "m"
  else:
    mode_cython = "s"
  
  cdef np.ndarray[double, ndim=1] ccf_peak_mc = np.zeros(nsim)
  cdef np.ndarray[double, ndim=1] tau_peak_mc = np.zeros(nsim)
  cdef np.ndarray[double, ndim=1] tau_cent_mc = np.zeros(nsim)

  cdef double *t1_cython = <double *>PyMem_Malloc(t1.shape[0]*sizeof(double))
  cdef double *f1_cython = <double *>PyMem_Malloc(t1.shape[0]*sizeof(double))
  cdef double *e1_cython = <double *>PyMem_Malloc(t1.shape[0]*sizeof(double))
  cdef double *t2_cython = <double *>PyMem_Malloc(t2.shape[0]*sizeof(double))
  cdef double *f2_cython = <double *>PyMem_Malloc(t2.shape[0]*sizeof(double))
  cdef double *e2_cython = <double *>PyMem_Malloc(t2.shape[0]*sizeof(double))

  cdef double *ccf_peak_mc_cython = <double *>PyMem_Malloc(nsim*sizeof(double))
  cdef double *tau_peak_mc_cython = <double *>PyMem_Malloc(nsim*sizeof(double))
  cdef double *tau_cent_mc_cython = <double *>PyMem_Malloc(nsim*sizeof(double))
  
  for i in range(t1.shape[0]):
    t1_cython[i] = t1[i]
    f1_cython[i] = f1[i]
    e1_cython[i] = e1[i]
  
  for i in range(t2.shape[0]):
    t2_cython[i] = t2[i]
    f2_cython[i] = f2[i]
    e2_cython[i] = e2[i]

  # call c function  
  ciccf_mc_oneway(t1_cython, f1_cython, e1_cython, t1.shape[0],
           t2_cython, f2_cython, e2_cython, t2.shape[0],
           ntau, tau_beg, tau_end, nsim, 
           threshold, mode_cython, ignore_warning,
           ccf_peak_mc_cython, tau_peak_mc_cython, tau_cent_mc_cython)
  
  for i in range(nsim):
    ccf_peak_mc[i] = ccf_peak_mc_cython[i]
    tau_peak_mc[i] = tau_peak_mc_cython[i]
    tau_cent_mc[i] = tau_cent_mc_cython[i]

  PyMem_Free(t1_cython)
  PyMem_Free(f1_cython)
  PyMem_Free(e1_cython)
  PyMem_Free(t2_cython)
  PyMem_Free(f2_cython)
  PyMem_Free(e2_cython)

  PyMem_Free(ccf_peak_mc_cython)
  PyMem_Free(tau_peak_mc_cython)
  PyMem_Free(tau_cent_mc_cython)
  return ccf_peak_mc, tau_peak_mc, tau_cent_mc
