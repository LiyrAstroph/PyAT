#===================================================================================#
#  PyAT: Python Astronomical Tools
#  A package providing basic, common tools in astronomical analysis
#
#  Yan-Rong Li, liyropt@gmail.com
#  2023-08-31
#===================================================================================#

__all__ = ["format_mica", "load_mica"]

import numpy as np 

def format_mica(fname, data1, data2):
  """
  This function generates a formatted file for MICA.
  If data1 is none, the vmap mode in MICA is presumed.
  If data1 is a list, each entity represets one driving light curve,
  the corresponding responding light curves are stored in data2, 
  with the same order. 

  Parameters
  ----------
  fname : string
    File name
  
  data1 : 2D array like
    The driving light curves, time, flux, and error.
  
  data2 : 2D array like
    The responding light curves, time, flux, and error.
  
  Returns
  -------
  None : None
    No returns.
  """
  
  # data1 is none
  if data1 is None:
    if not isinstance(data2, list):
      fp = open(fname, "w")
      fp.write("# 1\n")
      fp.write("# %d:%d\n"%(0, data2.shape[0]))
      np.savetxt(fp, data2, fmt="%f")
      fp.close()
    else:
      fp = open(fname, "w")
      fp.write("# 1\n")
      fp.write("# %d"%0)
      for i in range(len(data2)):
        fp.write(":%d"%data2[i].shape[0])
      fp.write("\n")
      for i in range(len(data2)):
        np.savetxt(fp, data2[i], fmt="%f")
        fp.write("\n")
      
      fp.close()

  else:
    # data1 has only one set
    if not isinstance(data1, list):
      if not isinstance(data2, list):
        fp = open(fname, "w")
        fp.write("# 1\n")
        fp.write("# %d:%d\n"%(data1.shape[0], data2.shape[0]))
        np.savetxt(fp, data1, fmt="%f")
        fp.write("\n")
        np.savetxt(fp, data2, fmt="%f")
        fp.close()
      else:
        fp = open(fname, "w")
        fp.write("# 1\n")
        fp.write("# %d"%data1.shape[0])
        for i in range(len(data2)):
          fp.write(":%d"%data2[i].shape[0])
        fp.write("\n")
        np.savetxt(fp, data1, fmt="%f")
        for i in range(len(data2)):
          fp.write("\n")
          np.savetxt(fp, data2[i], fmt="%f")
        
        fp.close()
    else:  # data1 has multiple sets
      fp = open(fname, "w")
  
      # write header
      fp.write("# %d\n"%len(data1))
      for i in range(len(data1)):
        fp.write("# %d"%data1[i].shape[0])
        if not isinstance(data2[i], list):
          fp.write(":%d\n"%data2[i].shape[0])
        else:
          for j in range(len(data2[i])):
            fp.write(":%d"%data2[i][j].shape[0])
          fp.write("\n")
      # write data
      ic = 0
      for i in range(len(data1)):
        if ic != 0:
          fp.write("\n")
        np.savetxt(fp, data1[i], fmt="%f")
        if not isinstance(data2[i], list):
          fp.write("\n")
          np.savetxt(fp, data2[i], fmt="%f")
        else:
          for j in range(len(data2[i])):
            fp.write("\n")
            np.savetxt(fp, data2[i][j], fmt="%f")
        
        ic += 1
      
      fp.close()

def load_mica(fname):
  """
  Load mica-format data
  
  Parameters
  ----------
  fname : str 
    file name
  
  Returns
  -------
  data : dict
    loaded data from input file
  """

  fp = open(fname, "r")
  
  line = fp.readline()
  nd = int(line[1:])

  ns = []
  for i in range(nd):
    line = fp.readline()
    ns.append(line[1:].split(":"))

  data = {}
  for i in range(nd):
    data["set%d"%i] = {}
    for j in range(len(ns[i])):
      datalc = np.zeros((int(ns[i][j]), 3))
      for k in range(int(ns[i][j])):
        line = fp.readline()
        datalc[k, :] = line.split()
      fp.readline()
      data["set%d"%i]["lc%d"%j] = datalc 
  fp.close()

  return data
