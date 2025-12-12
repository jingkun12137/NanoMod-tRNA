#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for NanoMod-tRNA package
"""

from setuptools import setup, find_packages

setup(
    name="NanoMod-tRNA",
    version="0.9.6",
    description="NanoMod-tRNA: Attention MIL with Adaptive Training Strategy for tRNA modification detection",
    author="Yi Jingkun",
    author_email="1810305301@pku.edu.cn",
    url="https://github.com/jingkun12137/NanoMod-tRNA",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.23.0",
        "datatable>=0.11.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pysam>=0.16.0",
    ],
    entry_points={
        'console_scripts': [
            'NanoMod=NanoMod.__main__:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",
)
