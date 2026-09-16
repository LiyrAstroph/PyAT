from math import tau

import numpy as np
import pyat
import matplotlib.pyplot as plt 

import piccf_mc

def test_syserr():
  data = np.loadtxt("lightcurve_example.txt")

  syserr = pyat.estimate_syserr(data[:, 0], data[:, 1], data[:, 2], size=2)

  print("syserr:", syserr)
  plt.errorbar(data[:, 0], data[:, 1], yerr=np.sqrt(data[:, 2]**2+syserr**2), ls='none')
  plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], ls='none', capsize=1.5)
  plt.show()

def test_line_widths():

  wave = np.linspace(4700, 4960.0, 101)
  flux = 2.2*np.exp(-0.5*(wave - 4850.0)**2/20.0**2) + np.exp(-0.5*(wave - 4760.0)**2/10.0**2)

  wave_win = [wave[0], wave[-1]]
  wl, wr, wmax, fmax, sigma = pyat.get_line_widths(wave, flux, line_win=wave_win, return_full=True, doplot=True)
  
  print("width:", wr-wl, "sigma:", sigma)
  plt.show()
  # plt.plot(wave, flux)
  # plt.axhline(y=fmax*0.5, ls='--')
  # plt.axvline(x=wl, ls='--')
  # plt.axvline(x=wr, ls='--')
  # plt.show()

def test_rebin():
  import spectres

  wave = np.linspace(4700, 4960.0, 101)
  flux = 1.2*np.exp(-0.5*(wave - 4850.0)**2/20.0**2) + np.exp(-0.5*(wave - 4760.0)**2/10.0**2)

  wave = np.delete(wave, np.arange(20, 30))
  flux = np.delete(flux, np.arange(20, 30))

  wave_rebin = np.linspace(4700, 4980.0, 101)
  wave_rebin = np.delete(wave_rebin, np.arange(20, 30))
  flux_rebin = pyat.rebin_spectrum(wave_rebin, wave, flux)
  
  #fs = spectres.spectres(wave_rebin, wave, flux)

  wave_edge = pyat.get_bin_edge(wave)
  x = np.array(list(zip(wave_edge[:-1], wave_edge[1:]))).flatten()
  y = np.array(list(zip(flux, flux))).flatten()
  plt.plot(x, y, color="C1")

  wave_rebin_edge = pyat.get_bin_edge(wave_rebin)
  x = np.array(list(zip(wave_rebin_edge[:-1], wave_rebin_edge[1:]))).flatten()
  y = np.array(list(zip(flux_rebin, flux_rebin))).flatten()
  plt.plot(x, y, color='C2')
  plt.plot(wave, flux, marker='o', label='Data', ls='none', color="C1")
  plt.plot(wave_rebin, flux_rebin, marker='o', label='PyAT Rebin', ls='none', color="C2")
  #plt.plot(wave_rebin, fs, label='Spectres')
  plt.legend()
  plt.show()

def test_rebin_error():
  import spectres

  wave, flux, err = np.loadtxt("spectrum_example.txt", usecols=(0,1,2), unpack=True)

  wave_rebin = np.linspace(wave[0], wave[-1], 200)
  #wave_rebin = wave
  flux_rebin, err_rebin = pyat.rebin_spectrum_with_error(wave_rebin, wave, flux, err)
  
  fs,es = spectres.spectres(wave_rebin, wave, flux, err)
  
  fig = plt.figure()
  ax = fig.add_subplot(211)
  plt.plot(wave, flux)
  plt.plot(wave_rebin, flux_rebin)
  plt.plot(wave_rebin, fs)
  ax.set_ylabel("Flux")
  ax = fig.add_subplot(212)
  plt.plot(wave, err, label='data')
  plt.plot(wave_rebin, err_rebin, label='PyAT Rebin')
  plt.plot(wave_rebin, es, label='Spectres Rebin')
  ax.legend()
  ax.set_ylabel("Error")
  plt.show()

def test_ccf():
  cont = np.loadtxt("lightcurve_echo_example1.txt")
  line = np.loadtxt("lightcurve_echo_example2.txt")

  cont_detrend = pyat.detrend(cont[:, 0], cont[:, 1], cont[:, 2], order=1)
  line_detrend = pyat.detrend(line[:, 0], line[:, 1], line[:, 2], order=1)

  pyat.iccf_prmax_null(cont[:, 0], cont_detrend, cont[:, 2], line[:, 0], line_detrend, line[:, 2],
                      1001, -50.0, 100.0, gapx=None, gapy=None, doplot=True)
  
  # estimate iccf peak significance
  pyat.iccf_peak_significance(cont[:, 0], cont_detrend, cont[:, 2], line[:, 0], line_detrend, line[:, 2],
                              1001, -50.0, 100.0, 1000, doshow=True)
  return
  
  fig = plt.figure(1)
  ax = fig.add_subplot(121)
  plt.errorbar(cont[:, 0], cont[:, 1], yerr=cont[:, 2], ls='none')
  plt.errorbar(line[:, 0], line[:, 1]*5, yerr=line[:, 2]*5, ls='none')
  
  ax = fig.add_subplot(122)
  t, r, rmax, tau_peak, tau_cent = pyat.iccf(cont[:, 0], cont[:, 1], line[:, 0], line[:, 1], 
                                                   1001, -50.0, 100, threshold=0.8, mode='single')
  print(rmax, tau_peak, tau_cent)
  plt.plot(t, r)

  t, r, rmax, tau_peak, tau_cent = pyat.iccf_slow(cont[:, 0], cont[:, 1], line[:, 0], line[:, 1], 
                                                   1001, -50.0, 100, threshold=0.8, mode='single')
  plt.plot(t, r)

  t, r, rmax, tau_peak, tau_cent = piccf_mc.piccf(cont[:, 0], cont[:, 1], line[:, 0], line[:, 1], 
                                                   1001, -50.0, 100)
  print(rmax, tau_peak, tau_cent)
  plt.plot(t, r)

  plt.axhline(y=rmax*0.8, ls='--')
  ax.set_xlabel("Time Lag")
  ax.set_ylabel("ICCF")
  plt.show()
  
  fig = plt.figure(1)
  ax1 = fig.add_subplot(311)
  ax2 = fig.add_subplot(312)
  ax3 = fig.add_subplot(313)

  rmax_mc, tau_peak_mc, tau_cent_mc = pyat.iccf_mc(cont[:, 0], cont[:, 1], cont[:, 2], line[:, 0], line[:, 1], line[:, 2], 
                                                   500, -50.0, 100.0, threshold=0.8, mode="multiple", nsim=2000, ignore_warning=True)

  ax1.hist(tau_peak_mc, bins=30, label='centroid', range=[0, 60])
  ax2.hist(tau_cent_mc, bins=30, label='peak', range=[0, 60])
  ax3.hist(rmax_mc, bins=30, range=[0.4, 1.0])

  rmax_mc, tau_peak_mc, tau_cent_mc = pyat.iccf_mc_slow(cont[:, 0], cont[:, 1], cont[:, 2], line[:, 0], line[:, 1], line[:, 2], 
                                                   500, -50.0, 100.0, threshold=0.8, mode="multiple", nsim=2000, ignore_warning=True)
  
  ax1.hist(tau_peak_mc, bins=30, label='centroid', alpha=0.5, range=[0, 60])
  ax2.hist(tau_cent_mc, bins=30, label='peak', alpha=0.5, range=[0, 60])
  ax3.hist(rmax_mc, bins=30, alpha=0.5, range=[0.4, 1.0])

  tau_cent_mc, tau_peak_mc = piccf_mc.piccf_mc(cont[:, 0], cont[:, 1], cont[:, 2], line[:, 0], line[:, 1], line[:, 2], 
                                                   500, -50.0, 100.0, 2000)
  ax1.hist(tau_peak_mc, bins=30, label='centroid', alpha=0.5, range=[0, 60])
  ax2.hist(tau_cent_mc, bins=30, label='peak', alpha=0.5, range=[0, 60])
  plt.show()
  
  # estimate iccf peak significance
  pyat.iccf_peak_significance(cont[:, 0], cont[:, 1], cont[:, 2], line[:, 0], line[:, 1], line[:, 2],
                              1001, 0, 100, 1000, doshow=True)
  
  # test one-way iccf
  fig = plt.figure(1)
  ax = fig.add_subplot(121)
  plt.errorbar(cont[:, 0], cont[:, 1], yerr=cont[:, 2], ls='none')
  plt.errorbar(line[:, 0], line[:, 1]*5, yerr=line[:, 2]*5, ls='none')
  
  ax = fig.add_subplot(122)
  t, r, rmax, tau_peak, tau_cent = pyat.iccf_oneway(cont[:, 0], cont[:, 1], line[:, 0], line[:, 1], 
                                                   1001, -50.0, 100, threshold=0.8, mode='single')
  print(rmax, tau_peak, tau_cent)
  plt.plot(t, r)

  t, r, rmax, tau_peak, tau_cent = pyat.iccf_oneway_slow(cont[:, 0], cont[:, 1], line[:, 0], line[:, 1], 
                                                   1001, -50.0, 100, threshold=0.8, mode='single')
  
  print(rmax, tau_peak, tau_cent)
  plt.plot(t, r)
  plt.axhline(y=rmax*0.8, ls='--')
  ax.set_xlabel("Time Lag")
  ax.set_ylabel("ICCF")
  plt.show()

  pyat.iccf_oneway_peak_significance(cont[:, 0], cont[:, 1], cont[:, 2], line[:, 0], line[:, 1], line[:, 2],
                              1001, 0, 100, 1000, doshow=True)
  
  fig = plt.figure(1)
  ax1 = fig.add_subplot(311)
  ax2 = fig.add_subplot(312)
  ax3 = fig.add_subplot(313)

  rmax_mc, tau_peak_mc, tau_cent_mc = pyat.iccf_mc_oneway(cont[:, 0], cont[:, 1], cont[:, 2], line[:, 0], line[:, 1], line[:, 2], 
                                                   500, -50.0, 100.0, threshold=0.8, mode="single", nsim=2000, ignore_warning=True)

  ax1.hist(tau_peak_mc, bins=30, label='centroid', range=[0, 60])
  ax2.hist(tau_cent_mc, bins=30, label='peak', range=[0, 60])
  ax3.hist(rmax_mc, bins=30, range=[0.4, 1.0])

  rmax_mc, tau_peak_mc, tau_cent_mc = pyat.iccf_mc_oneway_slow(cont[:, 0], cont[:, 1], cont[:, 2], line[:, 0], line[:, 1], line[:, 2], 
                                                   500, -50.0, 100.0, threshold=0.8, mode="single", nsim=2000, ignore_warning=True)
  
  ax1.hist(tau_peak_mc, bins=30, label='centroid', alpha=0.5, range=[0, 60])
  ax2.hist(tau_cent_mc, bins=30, label='peak', alpha=0.5, range=[0, 60])
  ax3.hist(rmax_mc, bins=30, alpha=0.5, range=[0.4, 1.0])
  plt.show()

def test_detrend():
  
  data = np.loadtxt("lightcurve_echo_example1.txt")
  yd, trend=pyat.detrend(data[:, 0], data[:, 1], data[:, 2], order=2, return_trend=True)
  
  plt.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], ls='none')
  plt.errorbar(data[:, 0], yd, yerr=data[:, 2], ls='none')
  x = data[:, 0] - (data[0, 0]+data[-1, 0])/2
  y = trend[0] + trend[1] * x + trend[2] * x**2
  plt.plot(data[:, 0], y, ls='--')
  plt.show()

def test_loadtemplate():
  from pyat import load_template
  
  template = load_template("AGN_NIR")
  
  plt.plot(template[:, 0], template[:, 1], label='AGN NIR')
  template = load_template("AGN_SDSS")
  print(template)
  plt.show() 

  plt.plot(template[:, 0], template[:, 1], label='AGN SDSS')
  plt.show()
  
def ndeff_estimate(n, dt, taux, tauy, errx, sigx, erry, sigy):
    sum = 0.0
    for i in range(1, n):
        sum += (1.0 - i/n) * np.exp(-dt*i/taux)/(1.0+errx**2/sigx**2) * np.exp(-dt*i/tauy)/(1.0+erry**2/sigy**2)
    return n/(1.0 + 2.0*sum)

def ccf_nd(t1, y1, ye1, t2, y2, ye2, ntau, tau_beg, tau_end, sig1, taud1, sig2, taud2):
    """
    count the number of points in each time lag bins
    """
    taud12 = taud1*taud2/(taud1+taud2)

    tau = np.linspace(tau_beg, tau_end, ntau)
    
    nd1 = np.zeros(ntau)
    nd2 = np.zeros(ntau)
    ndeff1 = np.zeros(ntau)
    ndeff2 = np.zeros(ntau)

    for i in range(ntau):
        taui = tau[i]
        
        # first interpolate y1
        idx = np.where((t2-taui>=t1[0])&(t2-taui<=t1[-1]))[0]
        t2_new = t2[idx]
        y2_new = y2[idx]
        ye2_new = ye2[idx]
        y1_new = np.interp(t2_new, t1, y1)
        ye1_new = np.interp(t2_new, t1, ye1)
        nd1[i] = t2_new.shape[0]
        
        err1 = np.mean(ye1_new)
        err2 = np.mean(ye2_new)
        gap = np.max(t2_new[1:]-t2_new[:-1])
        if gap < 20:
            gap = 0
        dt = (t2_new[-1]-t2_new[0]-gap)/(t2_new.shape[0]-1)
        # ndeff1[i] = (t2_new.shape[0])/(1+2/(np.exp(dt/taud12)-1)/(1.0+err1**2/sig1**2)/(1.0+err2**2/sig2**2))
        ndeff1[i] = ndeff_estimate(t2_new.shape[0], dt, taud1, taud2, err1, sig1, err2, sig2)

        # then interpolat y2
        idx = np.where((t1+taui>=t2[0])&(t1+taui<=t2[-1]))[0]
        t1_new = t1[idx]
        y1_new = y1[idx]
        ye1_new = ye1[idx]
        y2_new = np.interp(t1_new, t2, y2)
        ye2_new = np.interp(t1_new, t2, ye2)
        nd2[i] = t1_new.shape[0]

        err1 = np.mean(ye1_new)
        err2 = np.mean(ye2_new)
        gap = np.max(t1_new[1:]-t1_new[:-1])
        if gap < 35:
            gap = 0.0
        dt = (t1_new[-1]-t1_new[0]-gap)/(t1_new.shape[0]-1)
        # ndeff2[i] = (t1_new.shape[0])/(1+2/(np.exp(dt/taud12)-1)/(1.0+err1**2/sig1**2)/(1.0+err2**2/sig2**2))
        ndeff2[i] = ndeff_estimate(t1_new.shape[0], dt, taud1, taud2, err1, sig1, err2, sig2)

        # print(taui, y1_new.shape[0], y2_new.shape[0])

    return tau, nd1, ndeff1, nd2, ndeff2

def test_sim_drw():
  data1 = np.loadtxt("sim1.txt")
  sample1, prob1 = pyat.drw_modeling(data1[:, 0], data1[:, 1], data1[:, 2], doshow=True, return_prob=True)
  idx = (sample1[:, 1]<np.log(200/10.0))
  sample1 = sample1[idx, :]
  prob1 = prob1[idx]

  data2 = np.loadtxt("sim2.txt")
  sample2, prob2 = pyat.drw_modeling(data2[:, 0], data2[:, 1], data2[:, 2], doshow=True, return_prob=True)
  idx = (sample2[:, 1]<np.log(200/10.0))
  sample2 = sample2[idx, :]
  prob2 = prob2[idx]

  t_data, r_data, rmax_data, tau_peak = pyat.iccf_peak(data1[:, 0], data1[:, 1], data2[:, 0], data2[:, 1],
                                                   501, -40.0, 40.0)
  nsim = 1000
  z_all = np.zeros((nsim, 501))
  zmax_all = np.zeros(nsim)
  tauxy_all = np.zeros(nsim)
  z_grid = np.linspace(-2, 2, 500)
  pdf_grid = np.zeros(len(z_grid))
  for i in range(nsim):
    sigma1, tau1 = np.exp(sample1[np.random.randint(0, len(sample1)), :])
    fs, fe = pyat.genlc_psd_drw_data([sigma1, tau1], data1)
    sim1 = np.column_stack((data1[:, 0], fs, fe))

    sigma2, tau2 = np.exp(sample2[np.random.randint(0, len(sample2)), :])
    fs, fe = pyat.genlc_psd_drw_data([sigma2, tau2], data2)
    sim2 = np.column_stack((data2[:, 0], fs, fe))

    t, r, rmax, tau_peak = pyat.iccf_peak(sim1[:, 0], sim1[:, 1], sim2[:, 0], sim2[:, 1],
                                                   501, -40.0, 40)
    z_all[i, :] = np.arctanh(r)
    zmax_all[i] = np.arctanh(rmax)
    tauxy_all[i] = tau1*tau2/(tau1+tau2)

    # tau, nd1, ndeff1, nd2, ndeff2 = ccf_nd(sim1[:, 0], sim1[:, 1], sim1[:, 2], sim2[:, 0], sim2[:, 1], sim2[:, 2],
    #      501, -40.0, 40.0, sigma1, tau1, sigma2, tau2)
    # ndeff = (ndeff1+ndeff2)/2
    # std = 1.0/np.sqrt(ndeff[250])
    # pdf_grid += 1.0/np.sqrt(2*np.pi)/std*np.exp(-0.5*z_grid**2/std**2)

  sigma1, tau1 = np.exp(sample1[np.argmax(prob1)])
  sigma2, tau2 = np.exp(sample2[np.argmax(prob2)])
  tau, nd1, ndeff1, nd2, ndeff2 = ccf_nd(sim1[:, 0], sim1[:, 1], sim1[:, 2], 
                                         sim2[:, 0], sim2[:, 1], sim2[:, 2],
                                         501, -40.0, 40.0, sigma1, tau1, sigma2, tau2)
  ndeff = (ndeff1+ndeff2)/2
  print("ndeff:", ndeff[250])
  
  print(np.count_nonzero(zmax_all>np.arctanh(rmax_data))/nsim)

  fig = plt.figure()
  ax = fig.add_subplot(121)
  plt.hist(z_all[:, 250], bins=40, density=True)
  std = np.std(z_all[:, 250])
  print("ndeff:", 1.0/std**2)
  x = np.linspace(-2, 2, 500)
  pdf = 1.0/np.sqrt(2*np.pi*std**2)*np.exp(-0.5*x**2/std**2)
  plt.plot(x, pdf)
  pdf_grid /= np.sum(pdf_grid)*(z_grid[1]-z_grid[0])
  plt.plot(z_grid, pdf_grid)

  ax = fig.add_subplot(122)
  plt.hist(zmax_all, bins=40, density=True)
  plt.show()

  # estimate iccf peak significance
  #pyat.iccf_peak_significance(data1[:, 0], data1[:, 1], data1[:, 2], data2[:, 0], data2[:, 1], data2[:, 2],
  #                            501, 0, 100, 1000, doshow=True)

if __name__ == "__main__":

  # test_loadtemplate()

  # test_syserr()
  
  # test_line_widths()
  
  # test_rebin()

  # test_rebin_error()

  test_ccf()
  
  # test_detrend()

  # test_sim_drw()