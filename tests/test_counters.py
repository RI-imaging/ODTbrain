#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests tilted backpropagation algorithm
"""
from __future__ import division, print_function

import multiprocessing as mp
import numpy as np
import os
from os.path import abspath, basename, dirname, join, split, exists
import platform
from scipy.ndimage import rotate
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

from common_methods import create_test_sino_2d, create_test_sino_3d, create_test_sino_3d_tilted, cutout, get_test_parameter_set, write_results, get_results, normalize


def test_sum2d():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_2d(N=10)
    p = get_test_parameter_set(1)[0]
    # complex
    jmc = mp.Value("i", 0)
    jmm = mp.Value("i", 0)

    f = odtbrain._Back_2D.sum_2d(sino, angles,
                                         jmc=jmc,
                                         jmm=jmm,
                                         **p)
    
    assert jmc.value == jmm.value
    assert jmc.value != 0


def test_fmap2d():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_2d(N=10)
    p = get_test_parameter_set(1)[0]
    # complex
    jmc = mp.Value("i", 0)
    jmm = mp.Value("i", 0)

    f = odtbrain._Back_2D.fourier_map_2d(sino, angles,
                                         jmc=jmc,
                                         jmm=jmm,
                                         **p)
    
    assert jmc.value == jmm.value
    assert jmc.value != 0


def test_back2d():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_2d(N=10)
    p = get_test_parameter_set(1)[0]
    # complex
    jmc = mp.Value("i", 0)
    jmm = mp.Value("i", 0)

    f = odtbrain._Back_2D.backpropagate_2d(sino, angles, padval=0,
                                           jmc=jmc,
                                           jmm=jmm,
                                           **p)
    assert jmc.value == jmm.value
    assert jmc.value != 0


def test_back3d():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_3d(Nx=10, Ny=10)
    p = get_test_parameter_set(1)[0]
    # complex
    jmc = mp.Value("i", 0)
    jmm = mp.Value("i", 0)
    f = odtbrain._Back_3D.backpropagate_3d(sino, angles, padval=0,
                                           dtype=np.float64,
                                           jmc=jmc,
                                           jmm=jmm,
                                           **p)
    assert jmc.value == jmm.value
    assert jmc.value != 0


def test_back3d_tilted():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino, angles = create_test_sino_3d(Nx=10, Ny=10)
    p = get_test_parameter_set(1)[0]
    # complex
    jmc = mp.Value("i", 0)
    jmm = mp.Value("i", 0)
    f = odtbrain._Back_3D_tilted.backpropagate_3d_tilted(sino, angles, padval=0,
                                                         dtype=np.float64,
                                                         jmc=jmc,
                                                         jmm=jmm,
                                                         **p)
    assert jmc.value == jmm.value
    assert jmc.value != 0


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()