#===================================================================================#
#  PyAT: Python Astronomical Tools
#  A package providing basic, common tools in astronomical analysis
#
#  Yan-Rong Li, liyropt@gmail.com
#  2023-08-31
#===================================================================================#
import numpy as np
from pyat import rebin
import matplotlib.pyplot as plt

lc = np.loadtxt("lc1.txt")

t, f, e = rebin(lc[:, 0], lc[:, 1], lc[:, 2], 2)

# Plot the rebinned light curve
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

fig = plt.figure(figsize=(8, 4))
plt.errorbar(lc[:, 0], lc[:, 1], lc[:, 2], ls="none", marker="s", fillstyle="none", label='Data')
plt.errorbar(t, f, e, ls="none", marker="o", fillstyle="full", label='Rebininned')
plt.legend()
plt.xlabel("Time (days)")
plt.ylabel("Flux")
plt.show()
fig.savefig("lc_rebin.jpg", dpi=300)
plt.close()