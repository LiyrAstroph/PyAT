#===============================================
# Python Astronomical Tools
#
# Developed by Yan-Rong Li, liyanrong@ihep.ac.cn
# 2023-08-31
#===============================================

from setuptools import setup, find_packages
from setuptools.extension import Extension
from glob import glob
import os

# basedir = os.path.dirname(os.path.abspath(__file__))
# src = glob(os.path.join(basedir, "src/pyat", "*.py"))

# extensions = [
#     Extension(name="pyat", sources=src),
# ]

setup(
    name="pyat",
    version="0.0.0",
    author="Yan-Rong Li",
    packages=find_packages(where="src"),
    package_dir={'':'src'},
    # ext_modules = extensions,
    install_requires=["numpy","scipy","numba"],
)
