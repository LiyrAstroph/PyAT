#===================================================================================#
#  PyAT: Python Astronomical Tools
#  A package providing basic, common tools in astronomical analysis
#
#  Yan-Rong Li, liyropt@gmail.com
#  2023-08-31
#===================================================================================#

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from numpy import get_include
from glob import glob
import os

# use relative path (relative to setup.py)
ext_dir = "src/pyat"

extensions = cythonize([
    Extension(name="pyat.rebin", 
              sources=glob(os.path.join(ext_dir, "rebin.pyx"))
              ),
              
    Extension(name="pyat.ccf", 
              sources=[os.path.join(ext_dir, "ccf.pyx")]
                    + [os.path.join(ext_dir, "libccf.c")]
                    + glob(os.path.join(ext_dir, "gsl*.c")),
              depends=[os.path.join(ext_dir, "ccf.pxd")]
                    + [os.path.join(ext_dir, "libccf.h")]
                    + glob(os.path.join(ext_dir, "gsl*.h")),
              libraries=["c", "m"],
              include_dirs=[get_include()]
              ),
])

setup(
    name="pyat",
    version="0.1.0",
    author="Yan-Rong Li",
    packages={"pyat", "pyat.template"},
    package_dir={'pyat':'src/pyat', 'pyat.template':'template'},
    package_data={"pyat.template": ["*.txt"]},
    ext_modules = extensions,
    install_requires=["numpy","scipy","numba","celerite","corner","emcee","astropy"],
)
