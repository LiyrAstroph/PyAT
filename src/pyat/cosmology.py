
#===================================================================================#
#  PyAT: Python Astronomical Tools
#  A package providing basic, common tools in astronomical analysis
#
#  Yan-Rong Li, liyropt@gmail.com
#  2023-08-31
#===================================================================================#

import numpy as np
import astropy.cosmology as astro_cosmo
from astropy.cosmology import FlatLambdaCDM, Planck18
from astropy import units as u

__all__ = ["cosmology"]

class cosmology():
    def __init__(self, H0=Planck18.H0.value, Om0=Planck18.Om0):
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    
    def list_realizations(self):
        print(astro_cosmo.realizations.available)
        return 
    
    def set_cosmology(self, realization):
        import importlib
        astro_cosmo = importlib.import_module("astropy.cosmology")
        self.cosmo = getattr(astro_cosmo, realization)
        print("Use cosmology: %s"%realization)
        return
    
    def comoving_volume(self, z1=0, z2=1.0, dOmega=1.0):
        """
        calculating comoving_volume
        
        :param z1: redshift 1
        :param z2: redshift 2
        """ 
        dV = self.cosmo.comoving_volume(z2) - self.cosmo.comoving_volume(z1)
        
        dV = dV * dOmega/(4.0*np.pi) 
        return dV.value  # in Mpc