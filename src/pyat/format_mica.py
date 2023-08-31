__all__ = ["format_mica",]

import numpy as np 

def format_mica(fname, data1, data2):
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