#===============================================
# Python Astronomical Tools
#
# Developed by Yan-Rong Li, liyanrong@ihep.ac.cn
# 2023-08-31
#===============================================

from setuptools import setup, find_packages

setup(
    name="pyat",
    version="0.0.0",
    author="Yan-Rong Li",
    packages=find_packages(where="src"),
    package_dir={'':'src'},
    install_requires=["numpy","scipy"],
)
