#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests 1D angular weights.
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

from common_methods import create_test_sino_2d, get_test_parameter_set, write_results, get_results


def test_angle_offset():
    """
    Tests if things are still correct when there is a 2PI offset in the angles.
    """
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_2d()
    parameters = get_test_parameter_set(2)
    # reference
    r1 = []
    for p in parameters:
        f1 = odtbrain._Back_2D.backpropagate_2d(sino, angles, weight_angles=False, **p)
        r1.append(f1)
    # with offset
    angles[::2] += 2*np.pi*np.arange(angles[::2].shape[0])
    r2 = []
    for p in parameters:
        f2 = odtbrain._Back_2D.backpropagate_2d(sino, angles, weight_angles=False, **p)    
        r2.append(f2)
    # with offset and weights
    r3 = []
    for p in parameters:
        f3 = odtbrain._Back_2D.backpropagate_2d(sino, angles, weight_angles=True, **p)    
        r3.append(f3)
    assert np.allclose(np.array(r1).flatten().view(float),
                       np.array(r2).flatten().view(float))
    assert np.allclose(np.array(r2).flatten().view(float),
                       np.array(r3).flatten().view(float))


def test_angle_swap():
    """
    Test if everything still works, when angles are swapped.
    """
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_2d()
    # remove elements so that we can see that weighting works
    angles = angles[:-2]
    sino = sino[:-2,:]
    parameters = get_test_parameter_set(2)
    # reference
    r1 = []
    for p in parameters:
        f1 = odtbrain._Back_2D.backpropagate_2d(sino, angles, weight_angles=True, **p)
        r1.append(f1)
    # change order of angles
    order = np.argsort(angles % .5) 
    angles = angles[order]
    sino = sino[order,:]
    r2 = []
    for p in parameters:
        f2 = odtbrain._Back_2D.backpropagate_2d(sino, angles, weight_angles=True, **p)    
        r2.append(f2)
    assert np.allclose(np.array(r1).flatten().view(float),
                       np.array(r2).flatten().view(float))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    

    