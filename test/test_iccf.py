#===================================================================================#
#  PyAT: Python Astronomical Tools
#  A package providing basic, common tools in astronomical analysis
#
#  Yan-Rong Li, liyropt@gmail.com
#  2023-08-31
#===================================================================================#
import numpy as np 
import matplotlib.pyplot as plt 
import pyat

lc1 = np.loadtxt("lc1.txt")
lc2 = np.loadtxt("lc2.txt")

ntau = 1001
tau_beg = -20.0
tau_end = 50.0
threshold = 0.8
mode = "multiple"
ignore_warning = False

t, r, rmax, tau_peak, tau_cent = pyat.iccf(lc1[:, 0], lc1[:, 1], lc2[:, 0], lc2[:, 1], 
                                           ntau, tau_beg, tau_end)
print("rmax, tau_peak, tau_cent: %.2f, %.2f, %.2f" % (rmax, tau_peak, tau_cent))

# calculate iccf
tau, ccf, rmax, tau_peak, tau_cent = pyat.iccf(lc1[:, 0], lc1[:, 1], lc2[:, 0], lc2[:, 1], 
                                               ntau, tau_beg, tau_end, threshold=threshold, mode=mode, ignore_warning=ignore_warning)

# peroform Monte Carlo simulation to determine the time lag uncertainties
nsim = 10000
ccf_peak_mc, tau_peak_mc, tau_cent_mc = pyat.iccf_mc(lc1[:, 0], lc1[:, 1], lc1[:, 2], 
                                                     lc2[:, 0], lc2[:, 1], lc2[:, 2], ntau, tau_beg, tau_end, nsim=nsim, threshold=threshold, 
                                                     mode=mode, ignore_warning=ignore_warning)

# # perform significance testing of the iccf peak
prob, rmax_sim = pyat.iccf_peak_significance(lc1[:, 0], lc1[:, 1], lc1[:, 2], 
                                             lc2[:, 0], lc2[:, 1], lc2[:, 2], 
                                             ntau, tau_beg, tau_end, nsim=10000, 
                                             doshow=True)

tau_peak_err = np.percentile(tau_peak_mc, [15.85, 84.15])
tau_cent_err = np.percentile(tau_cent_mc, [15.85, 84.15])
print("Peak ICCF:", rmax)
print("Peak time lag: {:.2f} +{:.2f} -{:.2f}".format(tau_peak, tau_peak_err[1]-tau_peak, tau_peak-tau_peak_err[0]))
print("Centroid time lag: {:.2f} +{:.2f} -{:.2f}".format(tau_cent, tau_cent_err[1]-tau_cent, tau_cent-tau_cent_err[0]))

# plot the ICCF
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(131)
plt.plot(tau, ccf, label="ICCF")
ax.axvline(x=tau_peak, color="red", label="Peak", ls='--')
ax.axvline(x=tau_cent, color="blue", label="Centroid", ls='--')
ax.set_xlabel("Time Lag (days)")
ax.set_ylabel("ICCF")
ax.legend()
ax.set_title("ICCF")

# plot histogram of time lags from FR/RSSMonte Carlo simulations
ax = fig.add_subplot(132)
ax.hist(tau_peak_mc, label="Peak", bins=30)
ax.hist(tau_cent_mc, label="Centroid", alpha=0.5, bins=30)
ax.legend()
ax.set_ylabel("Count")
ax.set_xlabel("Time Lag (days)")
ax.set_title("Time Lags from FR/RSS Simulations")

# plot the significance testing result
ax = fig.add_subplot(133)
ax.hist(rmax_sim, bins=30)
ax.set_xlabel("rmax")
ax.set_ylabel("Significance")
ax.axvline(x=rmax, color="red", label="rmax", ls='--')
ax.legend()
ax.set_title("Significance Testing of rmax: p={:.2e}".format(prob))
plt.show()  
fig.savefig("iccf.jpg", dpi=300)

# null test using the method in Li & Wang 2026
tau, sigma, fig_sigma = pyat.iccf_sigma_null(lc1[:, 0], lc1[:, 1], lc1[:, 2], 
                    lc2[:, 0], lc2[:, 1], lc2[:, 2], 
                    ntau, tau_beg, tau_end, gapx=None, gapy=None, 
                    doshow=True)
fig_sigma.savefig("sigma_null.jpg", dpi=300)
