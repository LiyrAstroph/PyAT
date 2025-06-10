#===============================================
# Python Astronomical Tools
#
# Developed by Yan-Rong Li, liyanrong@ihep.ac.cn
# 2023-08-31
#===============================================

__all__ = ["load_template", "list_templates"]

import os 
import numpy as np
from importlib.resources import files
from astropy.io import ascii 

def load_template(template_name):
    """
    Load a template file from the 'template' directory.

    Parameters
    ----------
    template_name : str
        The name of the template file to load (without path).

    Returns
    -------
    str
        The content of the template file.
    """

    if template_name.lower() not in ["agn_nir", "agn_sdss"]:
        raise ValueError(f"Template '{template_name}' not found in the template directory.")
    
    if template_name.lower() == "agn_nir":
        template_fullname = "template_glikman2006.txt"
    elif template_name.lower() == "agn_sdss":
        template_fullname = "template_sdss_vandenberk.txt"

    #template_path = files('pyat.template').joinpath(template_fullname).open("r")
    template_path = os.path.join(os.path.dirname(__file__), "template", template_fullname)

    data = ascii.read(template_path, format="cds")
    
    temp = np.column_stack((data.columns[0].data, data.columns[1].data, data.columns[2].data))

    return temp 

def list_templates():   
    """
    List available templates in the 'template' directory.

    Returns
    -------
    list
        A list of available template names.
    """
    temp = ["AGN_NIR", "AGN_SDSS"]
    name = ["AGN Spectrum in NIR (Glikman et al. 2006)", 
            "AGN Spectrum from SDSS (Van Denberk et al. 2001)"]

    for i in range(len(temp)):
        print(f"{temp[i]}: {name[i]}")

    return temp