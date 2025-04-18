import numpy as np
import pyat
import matplotlib.pyplot as plt 

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
  
  fig = plt.figure(1)
  ax = fig.add_subplot(121)
  plt.errorbar(cont[:, 0], cont[:, 1], yerr=cont[:, 2], ls='none')
  plt.errorbar(line[:, 0], line[:, 1]*5, yerr=line[:, 2]*5, ls='none')
  
  
  
  ax = fig.add_subplot(122)
  t, r, rmax, tau_peak, tau_cent = pyat.iccf(cont[:, 0], cont[:, 1], line[:, 0], line[:, 1], 
                                                   1001, -20.0, 50, mode='single')
  
  plt.plot(t, r)
  ax.set_xlabel("Time Lag")
  ax.set_ylabel("ICCF")
  plt.show()
  
  rmax_mc, tau_peak_mc, tau_cent_mc = pyat.iccf_mc(cont[:, 0], cont[:, 1], cont[:, 2], line[:, 0], line[:, 1], line[:, 2], 
                                                   500, -50.0, 50.0, threshold=0.8, mode="single", nsim=1000, ignore_warning=True)
  fig = plt.figure(1)
  plt.hist(tau_peak_mc, bins=20, label='centroid')
  plt.hist(tau_cent_mc, alpha=0.8, bins=20, label='peak')
  plt.legend()
  plt.xlabel("Time Lag")
  fig = plt.figure(2)
  plt.hist(rmax_mc, bins=20)
  plt.xlabel("Rmax")
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

if __name__ == "__main__":
  
  test_syserr()
  
  test_line_widths()
  
  test_rebin()

  test_rebin_error()

  test_ccf()
  
  test_detrend()