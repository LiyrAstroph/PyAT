
__all__ = ["iccf", "iccf_mc"]

import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
#import piccf_mc

@njit
def iccf(t1, f1, t2, f2, ntau, tau_beg, tau_end, 
         threshold=0.8, mode="multiple",ignore_warning=False):
  """
  Interpolated CCF
  """
  if mode not in ["multiple", "single"]:
    raise ValueError("mode = %s is not recognized! use 'multiple' or 'single'!")

  tau = np.linspace(tau_beg, tau_end, ntau)
  ccf = np.zeros(ntau)
  
  if t2[0] > t1[-1] or t1[0] > t2[-1]:
    print("no overlap, set ccf to zero.")
    return tau, ccf, -10.0, 0.0, 0.0
  
  for i in range(ntau):
    taui = tau[i]

    # first interpolate f2
    idx = np.where((t1>=t2[0]-taui) & (t1<=t2[-1]-taui))[0]
    t1_intp = t1[idx]
    f1_intp = f1[idx]
    f2_intp = np.interp(t1_intp, t2-taui, f2)
    ccf12 = np.mean((f1_intp - np.mean(f1_intp))*(f2_intp - np.mean(f2_intp))) / (np.std(f1_intp) * np.std(f2_intp))

    # second interpolate f1
    idx = np.where((t2>=t1[0]+taui) & (t2<=t1[-1]+taui))[0]
    t2_intp = t2[idx]
    f2_intp = f2[idx]
    f1_intp = np.interp(t2_intp, t1+taui, f1)
    ccf21 = np.mean((f1_intp - np.mean(f1_intp))*(f2_intp - np.mean(f2_intp))) / (np.std(f1_intp) * np.std(f2_intp))

    # use average
    ccf[i] = 0.5*(ccf12+ccf21)
  
  # peak tau, if there are multiple occurence of peaks, using the rightmost one.
  imax = np.where(ccf == np.max(ccf))[0][-1]
  tau_peak = tau[imax]
  ccf_peak = ccf[imax]

  if ccf_peak < 0.0:
    print("negative ccf peak!")
    return tau, ccf, ccf_peak, tau_peak, 0.0
  
  # centrod tau
  # points larger than a threshold
  idx_above = np.where(ccf >= threshold*ccf_peak)[0]

  if not ignore_warning:
    if idx_above[0] == 0:
      # plt.plot(tau, ccf)
      # plt.axhline(y=ccf_peak*threshold, ls='--')
      # plt.show()
      raise ValueError("tau_beg is too large to cover the region with ccf>threshold*ccf_peak.")
    if idx_above[-1] == ntau-1:
      # plt.plot(tau, ccf)
      # plt.axhline(y=ccf_peak*threshold, ls='--')
      # plt.show()
      raise ValueError("tau_end is too small to cover the region with ccf>threshold*ccf_peak.")
    
  if mode == "multiple":

    tau_cent = np.sum(tau[idx_above] * ccf[idx_above])/np.sum(ccf[idx_above])
  
  else:
    
    # only one point
    if idx_above.shape[0] == 1:
      return tau, ccf, ccf_peak, tau_peak, tau_peak

    # first check if there are multiple peaks
    dtau_idx = np.zeros(idx_above.shape[0])
    dtau_idx[1:] = tau[idx_above[1:]] - tau[idx_above[:-1]]
    dtau_idx[0] = dtau_idx[1]
    dtau_idx_min = np.min(dtau_idx)

    # if there is a large gap
    idx_gap = np.where(dtau_idx > 3*dtau_idx_min)[0]

    if idx_gap.shape[0] > 0:  # multiply peaked, use the major peak

      imax_left_all  = np.where((ccf < threshold*ccf_peak) & (tau < tau_peak) )[0]
      if imax_left_all.shape[0] > 0:
        imax_left = imax_left_all[-1]
      else:
        imax_left = 0

      imax_right_all = np.where((ccf < threshold*ccf_peak) & (tau > tau_peak) )[0]
      if imax_right_all.shape[0] > 0:
        imax_right = imax_right_all[0]
      else:
        imax_right = ntau

      ccf_sum  = np.sum(ccf[imax_left+1:imax_right]*tau[imax_left+1:imax_right])
      ccf_norm = np.sum(ccf[imax_left+1:imax_right])

      # left cross point
      # tau_cross_left = np.interp(threshold*ccf_peak, ccf[imax_left:imax_left+2], tau[imax_left:imax_left+2])
      # ccf_sum  += threshold*ccf_peak * tau_cross_left
      # ccf_norm += threshold*ccf_peak

      # right cross point
      # tau_cross_right = np.interp(threshold*ccf_peak, ccf[imax_right:imax_right-2:-1], tau[imax_right:imax_right-2:-1])
      # ccf_sum  += threshold*ccf_peak * tau_cross_right
      # ccf_norm += threshold*ccf_peak

      tau_cent = ccf_sum/ccf_norm

    else:  # singlely peaked

      tau_cent = np.sum(tau[idx_above] * ccf[idx_above])/np.sum(ccf[idx_above])

  # plt.plot(tau, ccf, marker='o', markersize=2)
  # plt.axhline(y=ccf_peak)
  # plt.axhline(y=threshold*ccf_peak)
  # # plt.axvline(x=tau_cross_right)
  # # plt.axvline(x=tau_cross_left)
  # plt.axvline(x=tau[imax])

  # print(tau_peak, ccf_peak, tau_cent)

  # t, r, rmax, tau_cent, tau_peak = piccf_mc.piccf(t1, f1, t2, f2, ntau, tau_beg, tau_end)
  # print(tau_peak, rmax, tau_cent)
  
  # plt.plot(t, r)
  # plt.show()

  return tau, ccf, ccf_peak, tau_peak, tau_cent

def iccf_mc(t1, f1, e1, t2, f2, e2, ntau, tau_beg, tau_end, 
            threshold=0.8, mode="multiple", nsim=1000, ignore_warning=False):
  """
  do mc simulation
  """
  print("doing MC simulation, waiting for a while...")
  ccf_peak_mc, tau_peak_mc, tau_cent_mc = np.zeros((3, nsim))

  for i in range(nsim):
    if i%(nsim/10) == 0:
      print("%d%%-"%(100*i/nsim), end="", flush=True)

    # resample f1
    rand = np.random.randint(low=0, high=len(f1), size=len(f1))
    rand = np.sort(rand)
    idxs, counts = np.unique(rand, return_counts=True)

    t1_sim = t1[idxs]
    e1_sim = e1[idxs]/np.sqrt(counts) # reduce errors
    # add Gaussian noises
    f1_sim = f1[idxs] + e1_sim * np.random.randn(idxs.shape[0])

    # resample f2
    rand = np.random.randint(low=0, high=len(f2), size=len(f2))
    rand = np.sort(rand)
    idxs, counts = np.unique(rand, return_counts=True)

    t2_sim = t2[idxs]
    e2_sim = e2[idxs]/np.sqrt(counts) # reduce errors
    # add Gaussian noises
    f2_sim = f2[idxs] + e2_sim * np.random.randn(idxs.shape[0])

    # calculate ccf
    tau, ccf, ccf_peak_mc[i], tau_peak_mc[i], tau_cent_mc[i] = iccf(t1_sim, f1_sim, t2_sim, f2_sim, ntau, tau_beg, tau_end,
                                                                    threshold=threshold, mode=mode, ignore_warning=ignore_warning)
  
  print("done")

  # fig = plt.figure(1)
  # #plt.hist(tau_peak_mc, density=True)
  # plt.hist(tau_cent_mc, alpha=0.5, density=True, range=[-100, 500])
  # #plt.hist(ccf_peak_mc, density=True)
  # #plt.show()

  # #fig = plt.figure(2)
  # tau_cent_mc, tau_peak_mc = piccf_mc.piccf_mc(t1, f1, e1, t2, f2, e2, ntau, tau_beg, tau_end, nsim)
  # #plt.hist(tau_peak_mc, density=True)
  # plt.hist(tau_cent_mc, alpha=0.5, density=True, range=[-100, 500])
  # plt.show()
  
  # only use positive peaks
  idx = np.where(ccf_peak_mc > 0.0)[0]
  return ccf_peak_mc[idx], tau_peak_mc[idx], tau_cent_mc[idx]



