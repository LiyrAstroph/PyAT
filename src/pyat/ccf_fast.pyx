#!/usr/bin/python
#cython: initializedcheck=False, boundscheck=False, wraparound=False, cdivision=True, profile=False

#===================================================================================#
#  PyAT: Python Astronomical Tools
#  A package providing basic, common tools in astronomical analysis
#
#  Yan-Rong Li, liyropt@gmail.com
#  2023-08-31
#===================================================================================#

__all__ = ["iccf", "iccf_mc", "iccf_oneway", "iccf_mc_oneway",
           "iccf_peak_significance", "iccf_oneway_peak_significance"
          ]

import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from pyat.drw import drw_modeling, genlc_psd_drw

np.import_array()

#============================================================================
# proto for iccf
#============================================================================
cdef iccf_proto(
  np.ndarray[double, ndim=1] t1, 
  np.ndarray[double, ndim=1] f1, 
  np.ndarray[double, ndim=1] t2, 
  np.ndarray[double, ndim=1] f2, 
  int ntau, 
  double tau_beg, 
  double tau_end, 
  double threshold=0.8, 
  str mode="multiple", 
  bint ignore_warning=False,
  int ways=0):
  """
  Python interface for interpolated cross-correlation function 

  Parameters
  ----------
  t1 : numpy array
    time of first light curve
  f1 : numpy array 
    flux of first light curve
  t2 : numpy array
    time of second light curve
  f2 : numpy array 
    flux of second light curve
  ntau : int
    number of time lag point
  tau_beg : double 
    starting time lag
  tau_end : double
    ending time lag 
  threshold : double 
    fraction threshold of peak cross-correlation coefficient for computing centroid time lag
  mode : string 
    a string indicates whether using single peak or multiple peaks to compute centroid time lag 
  ignore_warning : bool
    whether ignore warnings
  ways : int 
    specifiy two-way or one-way iccf

  Returns
  -------
  tau : numpy array
       time lag array
  ccf : numpy array
       cross-correlation coefficient array
  rmax : double
       peak cross-correlation coefficient
  tau_peak : double
       peak time lag
  tau_cent : double
       centroid time lag
  """
  
  if t1.shape[0] != f1.shape[0]:
    raise ValueError("t1 and f1 should have the same size!")
  
  if t2.shape[0] != f2.shape[0]:
    raise ValueError("t2 and f2 should have the same size!")

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
  ciccf_proto(t1_cython, f1_cython, t1.shape[0], t2_cython, f2_cython, t2.shape[0],
        ntau, tau_beg, tau_end, threshold, mode_cython, ignore_warning, ways,
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

#============================================================================
# proto for iccf mc
#============================================================================
cdef iccf_mc_proto(
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
  bint ignore_warning=False,
  int ways=0):
  """
  Monte Carlo simulation using the FR/RSS method (Peterson et al. 1998, ApJ, PASP, 110, 660).
  
  Parameters
  ----------
  t1 : numpy array
    time of first light curve
  f1 : numpy array 
    flux of first light curve
  e1 : numpy array 
    error of first light curve
  t2 : numpy array
    time of second light curve
  f2 : numpy array 
    flux of second light curve
  e3 : numpy array 
    error of second light curve
  ntau : int
    number of time lag point
  tau_beg : double 
    starting time lag
  tau_end : double
    ending time lag 
  nsim : int 
    number of simulations
  threshold : double 
    fraction threshold of peak cross-correlation coefficient for computing centroid time lag
  mode : string 
    a string indicates whether using single peak or multiple peaks to compute centroid time lag 
  ignore_warning : bool
    whether ignore warnings
  ways : int 
    specifiy two-way or one-way iccf
  
  Returns
  -------
  ccf_peak_mc : numpy array 
    Monte Carlo sample of ccf peak 
  tau_peak_mc : numpy array 
    Monte Carlo sample of peak time lag 
  tau_cent_mc : numpy array 
    Monte Carlo sample of centroid time lag
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

  cdef int i
  
  for i in range(t1.shape[0]):
    t1_cython[i] = t1[i]
    f1_cython[i] = f1[i]
    e1_cython[i] = e1[i]
  
  for i in range(t2.shape[0]):
    t2_cython[i] = t2[i]
    f2_cython[i] = f2[i]
    e2_cython[i] = e2[i]

  # call c function  
  ciccf_mc_proto(t1_cython, f1_cython, e1_cython, t1.shape[0],
           t2_cython, f2_cython, e2_cython, t2.shape[0],
           ntau, tau_beg, tau_end, nsim, 
           threshold, mode_cython, ignore_warning, ways,
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

#============================================================================
# iccf
# directly call iccf_proto
#============================================================================
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
  bint ignore_warning=False,
  int ways=0):
  """
  Python interface for interpolated cross-correlation function 

  Parameters
  ----------
  t1 : numpy array
    time of first light curve
  f1 : numpy array 
    flux of first light curve
  t2 : numpy array
    time of second light curve
  f2 : numpy array 
    flux of second light curve
  ntau : int
    number of time lag point
  tau_beg : double 
    starting time lag
  tau_end : double
    ending time lag 
  threshold : double 
    fraction threshold of peak cross-correlation coefficient for computing centroid time lag
  mode : string 
    a string indicates whether using single peak or multiple peaks to compute centroid time lag 
  ignore_warning : bool
    whether ignore warnings
  ways : int 
    specifiy two-way or one-way iccf

  Returns
  -------
  tau : numpy array
       time lag array
  ccf : numpy array
       cross-correlation coefficient array
  rmax : double
       peak cross-correlation coefficient
  tau_peak : double
       peak time lag
  tau_cent : double
       centroid time lag
  """
  return iccf_proto(t1, f1, t2, f2, ntau, tau_beg, tau_end, 
                    threshold=threshold, mode=mode, ignore_warning=ignore_warning, ways=0)

#============================================================================
# iccf_mc
# directly call iccf_mc_proto
#============================================================================
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
  bint ignore_warning=False,
  int ways=0):
  """
  Do Monte Carlo simulation using the FR/RSS method (Peterson et al. 1998, ApJ, PASP, 110, 660).
  
  Parameters
  ----------
  t1 : numpy array
    time of first light curve
  f1 : numpy array 
    flux of first light curve
  e1 : numpy array 
    error of first light curve
  t2 : numpy array
    time of second light curve
  f2 : numpy array 
    flux of second light curve
  e3 : numpy array 
    error of second light curve
  ntau : int
    number of time lag point
  tau_beg : double 
    starting time lag
  tau_end : double
    ending time lag 
  nsim : int 
    number of simulations
  threshold : double 
    fraction threshold of peak cross-correlation coefficient for computing centroid time lag
  mode : string 
    a string indicates whether using single peak or multiple peaks to compute centroid time lag 
  ignore_warning : bool
    whether ignore warnings
  ways : int 
    specifiy two-way or one-way iccf
  
  Returns
  -------
  ccf_peak_mc : numpy array 
    Monte Carlo sample of ccf peak 
  tau_peak_mc : numpy array 
    Monte Carlo sample of peak time lag 
  tau_cent_mc : numpy array 
    Monte Carlo sample of centroid time lag
  """
  return iccf_mc_proto(t1, f1, e1, t2, f2, e2, ntau, tau_beg, tau_end,
                       nsim=nsim, threshold=threshold, mode=mode, 
                       ignore_warning=ignore_warning, ways=0)

#============================================================================
# iccf_oneway
# directly call iccf_proto
#============================================================================
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
  Python interface for one-way interpolated cross-correlation function.
  The first light curve is interpolated. 

  Parameters
  ----------
  t1 : numpy array
    time of first light curve
  f1 : numpy array 
    flux of first light curve
  t2 : numpy array
    time of second light curve
  f2 : numpy array 
    flux of second light curve
  ntau : int
    number of time lag point
  tau_beg : double 
    starting time lag
  tau_end : double
    ending time lag 
  threshold : double 
    fraction threshold of peak cross-correlation coefficient for computing centroid time lag
  mode : string 
    a string indicates whether using single peak or multiple peaks to compute centroid time lag 
  ignore_warning : bool
    whether ignore warnings

  Returns
  -------
  tau : numpy array
       time lag array
  ccf : numpy array
       cross-correlation coefficient array
  rmax : double
       peak cross-correlation coefficient
  tau_peak : double
       peak time lag
  tau_cent : double
       centroid time lag
  """
  return iccf_proto(t1, f1, t2, f2, ntau, tau_beg, tau_end, 
                    threshold=threshold, mode=mode, ignore_warning=ignore_warning, ways=1)

#============================================================================
# iccf mc oneway
# directly call iccf_mc_proto
#============================================================================
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
  Do Monte Carlo simulation using the FR/RSS method (Peterson et al. 1998, ApJ, PASP, 110, 660).
  Only do interpolation on the driving light curve.

  Parameters
  ----------
  t1 : numpy array
    time of first light curve
  f1 : numpy array 
    flux of first light curve
  e1 : numpy array 
    error of first light curve
  t2 : numpy array
    time of second light curve
  f2 : numpy array 
    flux of second light curve
  e3 : numpy array 
    error of second light curve
  ntau : int
    number of time lag point
  tau_beg : double 
    starting time lag
  tau_end : double
    ending time lag 
  nsim : int 
    number of simulations
  threshold : double 
    fraction threshold of peak cross-correlation coefficient for computing centroid time lag
  mode : string 
    a string indicates whether using single peak or multiple peaks to compute centroid time lag 
  ignore_warning : bool
    whether ignore warnings
  
  Returns
  -------
  ccf_peak_mc : numpy array 
    Monte Carlo sample of ccf peak 
  tau_peak_mc : numpy array 
    Monte Carlo sample of peak time lag 
  tau_cent_mc : numpy array 
    Monte Carlo sample of centroid time lag
  """
  return iccf_mc_proto(t1, f1, e1, t2, f2, e2, ntau, tau_beg, tau_end,
                       nsim=nsim, threshold=threshold, mode=mode, 
                       ignore_warning=ignore_warning, ways=1)

#============================================================================
# proto for significance estimation of iccf peak
#============================================================================
cdef iccf_peak_significance_proto(
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
  int ways=0,
  bint doshow=False):
  """
  Significance testing of the iccf peak, that is, computing the probability
  for iccf peaks of mock light-curve pairs exceeding the iccf peak of input light curves.
  The mock light curves are assumed to be fully random and uncorrelated.

  Parameters
  ----------
  t1 : numpy array
    time of first light curve
  f1 : numpy array 
    flux of first light curve
  e1 : numpy array 
    error of first light curve
  t2 : numpy array
    time of second light curve
  f2 : numpy array 
    flux of second light curve
  e3 : numpy array 
    error of second light curve
  ntau : int
    number of time lag point
  tau_beg : double 
    starting time lag
  tau_end : double
    ending time lag 
  
  Returns
  -------
  prob : double 
    probability of iccf peaks exceeding the iccf peak of input light curves 
  rmax_sim : numpy array
    array of rmax of simulated light curves
  """
  if t1.shape[0] != f1.shape[0] or t1.shape[0] != e1.shape[0]:
    raise ValueError("t1, f1, and e1 should have the same size!")
  
  if t2.shape[0] != f2.shape[0] or t2.shape[0] != e2.shape[0]:
    raise ValueError("t2, f2, and e2 should have the same size!")
  
  cdef double prob=0.0
  cdef double sigma_drw, tau_drw
  cdef double dt1, dt2, dt, mu1_data, var1_data, mu2_data, var2_data, 
  cdef double mean_e1_data, mean_e2_data, std1_data_corr, std2_data_corr
  cdef double mu1, std1, mu2, std2
  cdef int i, j, ic, num1, num2, idx_max
  cdef double rmax, rmax_data, tau_peak, tau_peak_data

  cdef np.ndarray[double, ndim=2] sample1
  cdef np.ndarray[double, ndim=2] sample2
  cdef np.ndarray[int, ndim=1] ir1
  cdef np.ndarray[int, ndim=1] ir2
  
  cdef np.ndarray[double, ndim=1] ts
  cdef np.ndarray[double, ndim=1] fs
  cdef np.ndarray[double, ndim=1] fr1 = np.zeros(t1.shape[0])
  cdef np.ndarray[double, ndim=1] fe1 = np.zeros(t1.shape[0])
  cdef np.ndarray[double, ndim=1] fr2 = np.zeros(t2.shape[0])
  cdef np.ndarray[double, ndim=1] fe2 = np.zeros(t2.shape[0])
  cdef np.ndarray[double, ndim=1] rmax_sim = np.zeros(nsim)

  cdef double *tr1_cython = <double *>PyMem_Malloc(t1.shape[0]*sizeof(double)) 
  cdef double *fr1_cython = <double *>PyMem_Malloc(t1.shape[0]*sizeof(double))
  cdef double *tr2_cython = <double *>PyMem_Malloc(t2.shape[0]*sizeof(double))
  cdef double *fr2_cython = <double *>PyMem_Malloc(t2.shape[0]*sizeof(double))
  cdef double *tau_cython = <double *>PyMem_Malloc(ntau*sizeof(double))
  cdef double *ccf_cython = <double *>PyMem_Malloc(ntau*sizeof(double))
  cdef double *ccf_data_cython = <double *>PyMem_Malloc(ntau*sizeof(double))

  for i in range(t1.shape[0]):
    tr1_cython[i] = t1[i]
    fr1_cython[i] = f1[i]
  for i in range(t2.shape[0]):
    tr2_cython[i] = t2[i]
    fr2_cython[i] = f2[i]
  
  # rmax of the input data 
  ciccf_peak_proto(tr1_cython, fr1_cython, t1.shape[0], 
             tr2_cython, fr2_cython, t2.shape[0], 
             ntau, tau_beg, tau_end, ways,
             tau_cython, ccf_data_cython, &rmax_data, &idx_max, &tau_peak_data)

  # determine the minimum sampling interval
  # do not use minimum interval, because sometimes it is too tiny, 
  # leading to a huge number of points, very slow!
  # dt1 = np.min(t1[1:t1.shape[0]]-t1[0:t1.shape[0]-1])
  # dt2 = np.min(t2[1:t2.shape[0]]-t2[0:t2.shape[0]-1])
  dt1 = np.quantile(t1[1:t1.shape[0]]-t1[0:t1.shape[0]-1], q=0.01)
  dt2 = np.quantile(t2[1:t2.shape[0]]-t2[0:t2.shape[0]-1], q=0.01)
  if dt1 == 0:
    dt1 = t1[1]-t1[0]
    for i in range(2, t1.shape[0]):
      dt = t1[i]-t1[i-1]
      if  dt != 0 and dt < dt1:
        dt1 = dt 
  
  if dt2 == 0:
    dt2 = t2[1]-t2[0]
    for i in range(2, t2.shape[0]):
      dt = t2[i]-t2[i-1]
      if  dt != 0 and dt < dt2:
        dt2 = dt 

  num1 = int((t1[t1.shape[0]-1]-t1[0])/dt1)
  num2 = int((t2[t2.shape[0]-1]-t2[0])/dt2)

  # determine the mean and std 
  mu1_data = np.mean(f1)
  mu2_data = np.mean(f2)
  mean_e1_data = np.mean(e1**2)
  mean_e2_data = np.mean(e2**2)
  var1_data = np.var(f1, mean=mu1_data)
  var2_data = np.var(f2, mean=mu2_data)

  if(mean_e1_data >= var1_data): # variance is caused by noise
    std1_data_corr = 0.0  
  else:
    std1_data_corr = np.sqrt(var1_data - mean_e1_data)

  if(mean_e2_data >= var2_data): # variance is caused by noise
    std2_data_corr = 0.0
  else:
    std2_data_corr = np.sqrt(var2_data - mean_e2_data)

  # get DRW parameter sample 
  sample1 = drw_modeling(t1, f1, e1, doshow=doshow)
  sample2 = drw_modeling(t2, f2, e2, doshow=doshow)
  
  ir1 = np.random.randint(0, sample1.shape[0], size=nsim, dtype=np.int32)
  ir2 = np.random.randint(0, sample2.shape[0], size=nsim, dtype=np.int32)
  
  for i in range(nsim):
    if i%(nsim/10) == 0:
      print("%d%%-"%(100*i/nsim), end="")

    # generate mock light curve for f1
    sigma_drw, tau_drw = np.exp(sample1[ir1[i], :])
    ts, fs = genlc_psd_drw([sigma_drw, tau_drw, 1.0e-100], num1, dt1, 1.0e-10)
    ts += t1[0]

    fr1 = np.interp(t1, ts, fs)
    mu1  = np.mean(fr1)
    std1 = np.std(fr1)
    fr1 = (fr1-mu1)/std1 * std1_data_corr + mu1_data
    fe1 = abs(fr1) * abs(e1/f1)
    fr1 += np.random.randn(fr1.shape[0])*fe1

    # generate mock light curve for f1
    sigma_drw, tau_drw = np.exp(sample2[ir2[i], :])
    ts, fs = genlc_psd_drw([sigma_drw, tau_drw, 1.0e-100], num2, dt2, 1.0e-10)
    ts += t2[0]

    fr2  = np.interp(t2, ts, fs)
    mu2  = np.mean(fr2)
    std2 = np.std(fr2)
    fr2 = (fr2-mu2)/std2 * std2_data_corr + mu2_data
    fe2 = abs(fr2) * abs(e2/f2)
    fr2 += np.random.randn(fr2.shape[0])*fe2
    
    for j in range(t1.shape[0]):
      fr1_cython[j] = fr1[j]
    for j in range(t2.shape[0]):
      fr2_cython[j] = fr2[j]

    # call c function 
    ciccf_peak_proto(tr1_cython, fr1_cython, t1.shape[0], 
               tr2_cython, fr2_cython, t2.shape[0], 
               ntau, tau_beg, tau_end, ways,
               tau_cython, ccf_cython, &rmax, &idx_max, &tau_peak)
    
    rmax_sim[i] = rmax

    if doshow and i == 0:     
      import matplotlib.pyplot as plt
      fig = plt.figure(figsize=(12, 6))
      ax1 = fig.add_subplot(221)
      ax1.errorbar(t1, f1, yerr=e1)
      ax1.errorbar(t1, fr1, yerr=fe1)
      ax1.set_xlabel("Time")
      ax1.set_ylabel("Flux")
      xlim1 = ax1.get_xlim()

      ax2 = fig.add_subplot(223)
      ax2.errorbar(t2, f2, yerr=e2, label='data')
      ax2.errorbar(t2, fr2, yerr=fe2, label='mock', color='C1')
      ax2.set_xlabel("Time")
      ax2.set_ylabel("Flux")
      xlim2 = ax2.get_xlim()
      x1 = min(xlim1[0], xlim2[0])
      x2 = max(xlim1[1], xlim2[1])
      ax1.set_xlim(x1, x2)
      ax2.set_xlim(x1, x2)
      ax2.legend()

      ax = fig.add_subplot(122)

      tau = np.zeros(ntau)
      ccf = np.zeros(ntau)
      for j in range(ntau):
        tau[j] = tau_cython[j]
        ccf[j] = ccf_data_cython[j]
      ax.plot(tau, ccf, color='C0', label='data')

      for j in range(ntau):
        ccf[j] = ccf_cython[j]
      ax.plot(tau, ccf, color='C1', label='mock')

      ax.set_xlabel("Time Lag")
      ax.set_ylabel("ICCF")
      ax.legend()

      plt.show()
      
  print("done.")
  
  # counts number of rmax > rmax_data
  ic = 0
  for i in range(nsim):
    if rmax_sim[i] >= rmax_data:
      ic += 1

  prob = ic*1.0/nsim
  print("Prob: %.4f"%prob)

  if doshow:
    import matplotlib.pyplot as plt 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(rmax_sim, bins=30)
    ax.axvline(x=rmax_data, ls='--', color='red')
    ax.set_xlabel("r_max")
    ax.set_ylabel("Histogram")
    ax.set_title("%.3f"%prob)
    plt.show()

  PyMem_Free(tr1_cython)
  PyMem_Free(tr2_cython)
  PyMem_Free(fr1_cython)
  PyMem_Free(fr2_cython)
  PyMem_Free(tau_cython)
  PyMem_Free(ccf_cython)
  PyMem_Free(ccf_data_cython)

  return prob, rmax_sim
  
#============================================================================
# significiance estimation of two-way iccf peak
# directly call iccf_peak_significance_proto
#============================================================================
cpdef iccf_peak_significance(
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
  bint doshow=False):
  """
  Significance testing of the iccf peak, that is, computing the probability
  for iccf peaks of mock light-curve pairs exceeding the iccf peak of input light curves.
  The mock light curves are assumed to be fully random and uncorrelated.

  Parameters
  ----------
  t1 : numpy array
    time of first light curve
  f1 : numpy array 
    flux of first light curve
  e1 : numpy array 
    error of first light curve
  t2 : numpy array
    time of second light curve
  f2 : numpy array 
    flux of second light curve
  e3 : numpy array 
    error of second light curve
  ntau : int
    number of time lag point
  tau_beg : double 
    starting time lag
  tau_end : double
    ending time lag 
  
  Returns
  -------
  prob : double 
    probability of iccf peaks exceeding the iccf peak of input light curves 
  rmax_sim : numpy array
    array of rmax of simulated light curves
  """
  return iccf_peak_significance_proto(t1, f1, e1, t2, f2, e2, ntau, tau_beg, tau_end, 
                                      nsim=nsim, ways=0, doshow=doshow)

#============================================================================
# significiance estimation of one-way iccf peak
# directly call iccf_peak_significance_proto
#============================================================================
cpdef iccf_oneway_peak_significance(
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
  bint doshow=False):
  """
  Significance testing of the iccf peak, that is, computing the probability
  for iccf peaks of mock light-curve pairs exceeding the iccf peak of input light curves.
  The mock light curves are assumed to be fully random and uncorrelated.

  Parameters
  ----------
  t1 : numpy array
    time of first light curve
  f1 : numpy array 
    flux of first light curve
  e1 : numpy array 
    error of first light curve
  t2 : numpy array
    time of second light curve
  f2 : numpy array 
    flux of second light curve
  e3 : numpy array 
    error of second light curve
  ntau : int
    number of time lag point
  tau_beg : double 
    starting time lag
  tau_end : double
    ending time lag 
  
  Returns
  -------
  prob : double 
    probability of iccf peaks exceeding the iccf peak of input light curves 
  rmax_sim : numpy array
    array of rmax of simulated light curves
  """
  return iccf_peak_significance_proto(t1, f1, e1, t2, f2, e2, ntau, tau_beg, tau_end, 
                                      nsim=nsim, ways=1, doshow=doshow)