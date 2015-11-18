#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests tilted backpropagation algorithm
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
import odtbrain._br

from common_methods import create_test_sino_2d, create_test_sino_3d, cutout, get_test_parameter_set, write_results, get_results

WRITE_RES = False



def test_3d_backprop_phase_real():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_3d()
    parameters = get_test_parameter_set(2)
    # reference
    rref = list()
    for p in parameters:
        fref = odtbrain._Back_3D.backpropagate_3d(sino, angles, padval=0,
                                                  dtype=np.float64, onlyreal=True, **p)
        rref.append(cutout(fref))
    if WRITE_RES:
        write_results(myframe, rref)
    
    dataref = np.array(rref).flatten().view(float)
    
    
    r = list()
    for p in parameters:
        f = odtbrain._Back_3D_tilted.backpropagate_3d_tilted(sino, angles, padval=0,
                                               dtype=np.float64, onlyreal=True, **p)
        r.append(cutout(f))
    if WRITE_RES:
        write_results(myframe, r)
    data = np.array(r).flatten().view(float)
    
    assert np.allclose(data, dataref)

    

if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
