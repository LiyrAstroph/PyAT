__all__ = ["format_mica"]

import numpy as np 

def format_mica(fname, data1, data2):
  """
  generate formatted file for MICA.

  Parameters
  ----------
  fname : string
    File name
  
  data1 : 2D array like
    The driving light curve, time, flux, and error.
  
  data2 : 2D array like
    The responding light curve, time, flux, and error.
  
  Returns
  -------
  None : None
    No returns.
  """

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
  else:
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
