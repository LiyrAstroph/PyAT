#===============================================
# Python Astronomical Tools
#
# Developed by Yan-Rong Li, liyanrong@ihep.ac.cn
# 2023-08-31
#===============================================

__all__ = ["list_seds", "load_sed"]

import os 
import numpy as np
from importlib.resources import files
from astropy.io import ascii 

def list_seds():   
    """
    List available SEDs in the 'sed' directory.

    Returns
    -------
    list
        A list of available SED names.
    """
    sed = ["RQ_Shang2011", "RL_Shang2011"]
    name = ["Radio-quiet quasar SED (Shang et al. 2011)", 
            "Radio-loud quasar SED (Shang et al. 2011)"]

    for i in range(len(sed)):
        print(f"{sed[i]:<10} : {name[i]}")

    return sed

def load_sed(sed_name):
    """
    Load a SED file from the 'sed' directory.

    Parameters
    ----------
    sed_name : str
        The name of the SED file to load (without path).

    Returns
    -------
    str
        The content of the SED file.
    """

    if sed_name.lower() not in ["rq_shang2011", "rl_shang2011"]:
        raise ValueError(f"SED '{sed_name}' not found in the sed directory.")
    
    if sed_name.lower() == "rq_shang2011":
        sed_fullname = "sedrq_shang2011.txt"
    elif sed_name.lower() == "rl_shang2011":
        sed_fullname = "sedrl_shang2011.txt"

    #sed_path = files('pyat.sed').joinpath(sed_fullname).open("r")
    sed_path = os.path.join(os.path.dirname(__file__), "template", sed_fullname)

    data = ascii.read(sed_path, format="cds")

    sed = np.column_stack((data.columns[0].data, data.columns[1].data))

    return sed