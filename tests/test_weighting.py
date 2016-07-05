#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests backpropagation algorithm
"""
from __future__ import division, print_function

import numpy as np
import os
from os.path import abspath, basename, dirname, join, split, exists
import platform
import sys
import warnings
import zipfile

# Add parent directory to beginning of path variable
DIR = dirname(abspath(__file__))
sys.path = [split(DIR)[0]] + sys.path

import odtbrain
import odtbrain._Back_2D
import odtbrain._Back_3D
import odtbrain._Back_3D_tilted

from common_methods import create_test_sino_2d, create_test_sino_3d, cutout, get_test_parameter_set, write_results, get_results


def test_3d_backprop_weights_even():
    """
    even weights
    """
    platform.system = lambda:"Windows"
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_3d()
    p = get_test_parameter_set(1)[0]
    f1 = odtbrain._Back_3D.backpropagate_3d(sino, angles, weight_angles=False, **p)
    f2 = odtbrain._Back_3D.backpropagate_3d(sino, angles, weight_angles=True, **p)
    data1 = np.array(f1).flatten().view(float)
    data2 = np.array(f2).flatten().view(float)
    assert np.allclose(data1, data2)


def test_3d_backprop_tilted_weights_even():
    """
    even weights
    """
    platform.system = lambda:"Windows"
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_3d()
    p = get_test_parameter_set(1)[0]
    f1 = odtbrain._Back_3D_tilted.backpropagate_3d_tilted(sino, angles, weight_angles=False, **p)
    f2 = odtbrain._Back_3D_tilted.backpropagate_3d_tilted(sino, angles, weight_angles=True, **p)
    data1 = np.array(f1).flatten().view(float)
    data2 = np.array(f2).flatten().view(float)
    assert np.allclose(data1, data2)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    