#===============================================
# Python Astronomical Tools
#
# Developed by Yan-Rong Li, liyanrong@ihep.ac.cn
# 2023-08-31
#===============================================

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from numpy import get_include
from glob import glob
import os

basedir = os.path.dirname(os.path.abspath(__file__))

extensions = cythonize([
    Extension(name="pyat.rebin", 
              sources=glob(os.path.join(basedir, "src/pyat", "rebin.pyx"))
              ),
              
    Extension(name="pyat.ccf_fast", 
              sources=[os.path.join(basedir, "src/pyat", "ccf_fast.pyx")]
                    + [os.path.join(basedir, "src/pyat", "libccf.c")]
                    + glob(os.path.join(basedir, "src/pyat", "gsl*.c")),
              depends=[os.path.join(basedir, "src/pyat", "ccf_fast.pxd")]
                    + [os.path.join(basedir, "src/pyat", "libccf.h")]
                    + glob(os.path.join(basedir, "src/pyat", "gsl*.h")),
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
