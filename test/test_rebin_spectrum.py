import numpy as np
import matplotlib.pyplot as plt 

from pyat.rebin_spectrum import rebin_spectrum,            \
                                rebin_spectrum_with_error, \
                                get_bin_edge

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

def test_rebin():
  #import spectres

  wave = np.linspace(4700, 4960.0, 101)
  flux = 1.2*np.exp(-0.5*(wave - 4850.0)**2/20.0**2) + np.exp(-0.5*(wave - 4760.0)**2/10.0**2)

  wave = np.delete(wave, np.arange(20, 30))
  flux = np.delete(flux, np.arange(20, 30))

  wave_rebin = np.linspace(4700, 4980.0, 101)
  wave_rebin = np.delete(wave_rebin, np.arange(20, 30))
  flux_rebin = rebin_spectrum(wave_rebin, wave, flux)
  
  fs = spectres.spectres(wave_rebin, wave, flux)
  
  fig = plt.figure(figsize=(8, 4))
  ax = fig.add_subplot(111)
  # data
  wave_edge = get_bin_edge(wave)
  x = np.array(list(zip(wave_edge[:-1], wave_edge[1:]))).flatten()
  y = np.array(list(zip(flux, flux))).flatten()
  ax.plot(x, y, color="C1")
  ax.plot(wave, flux, marker='o', label='Data', ls='none', color="C1", markersize=4)

  # rebin
  wave_rebin_edge = get_bin_edge(wave_rebin)
  x = np.array(list(zip(wave_rebin_edge[:-1], wave_rebin_edge[1:]))).flatten()
  y = np.array(list(zip(flux_rebin, flux_rebin))).flatten()
  ax.plot(x, y, color='C2')
  ax.plot(wave_rebin, flux_rebin, marker='o', label='PyAT Rebin', color="C2", markersize=4)

  # # spectres
  # x = np.array(list(zip(wave_rebin_edge[:-1], wave_rebin_edge[1:]))).flatten()
  # y = np.array(list(zip(fs, fs))).flatten()
  # plt.plot(x, y, color="C3", ls=":")
  # plt.plot(wave_rebin, fs, marker='o', ls=':',label='Spectres', color="C3", markersize=4)
  ax.set_xlabel("Wavelength")
  ax.set_ylabel("Flux")
  ax.legend()
  ax.minorticks_on()
  plt.show()
  fig.savefig("spec_rebin1.jpg", dpi=300)

def test_rebin_error():
  #import spectres
  
  wave, flux, err = np.loadtxt("spectrum_example.txt", usecols=(0,1,2), unpack=True)
  wave_rebin = np.linspace(wave[0], wave[-1], 200)
  #wave_rebin = wave
  flux_rebin, err_rebin = rebin_spectrum_with_error(wave_rebin, wave, flux, err)
  
  #fs,es = spectres.spectres(wave_rebin, wave, flux, err)
  
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(211)
  plt.plot(wave, flux)
  plt.plot(wave_rebin, flux_rebin)
  #plt.plot(wave_rebin, fs)
  ax.set_ylabel("Flux")
  ax.minorticks_on()

  ax = fig.add_subplot(212)
  plt.plot(wave, err, label='data')
  plt.plot(wave_rebin, err_rebin, label='PyAT Rebin')
  #plt.plot(wave_rebin, es, label='Spectres Rebin')
  ax.minorticks_on()
  ax.legend()
  ax.set_ylabel("Error")
  ax.set_xlabel("Wavelength")
  plt.show()
  fig.savefig("spec_rebin2.jpg", dpi=300)


if __name__ == "__main__":
  test_rebin()
  test_rebin_error()