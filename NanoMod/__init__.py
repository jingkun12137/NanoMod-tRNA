#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NanoMod-tRNA: A deep learning tool for tRNA modification detection

This package provides tools for identifying and analyzing modifications in tRNA
sequences using deep learning techniques.
"""

__version__ = "0.9.6"
__author__ = "Yi Jingkun"
__email__ = "1810305301@pku.edu.cn"

# Core modules
from .model import NanoMod
from .train import train_model
from .predict import pure_predict
from .utils import (
    find_optimal_threshold,
    analyze_feature_importance,
    analyze_modification_patterns,
)

__all__ = [
    "train_model", "pure_predict", "NanoMod",
    "find_optimal_threshold", "analyze_feature_importance", "analyze_modification_patterns"
]
