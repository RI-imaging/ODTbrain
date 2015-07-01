#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests fourier_map algorithm
"""
from __future__ import division, print_function

import numpy as np
import os
from os.path import abspath, dirname, join, split
import sys


# Add parent directory to beginning of path variable
DIR = dirname(abspath(__file__))
sys.path = [split(DIR)[0]] + sys.path

import odtbrain._Back_2D


def get_2d_test_sinogram(A=5, N=13):
    sino = np.arange(A*N).reshape(A,N) * np.pi
    angles = np.linspace(0,1,A)
    return sino, angles


def get_test_parameter_set(set_number=1):
    res = 2.1
    lD = 0
    nm = 1.333
    parameters = []
    for i in range(set_number):
        parameters.append({"res" : res,
                           "lD" : lD,
                           "nm" : nm})
        res += .1
        lD += np.pi
        nm *= 1.01
    return parameters


def test_2d_fourier_map():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    sino, angles = get_2d_test_sinogram()
    parameters = get_test_parameter_set(2)
    r = list()
    for p in parameters:
        r.append(odtbrain._Back_2D.fourier_map_2d(sino, angles, **p))
    assert np.allclose(np.array(r).flatten().view(float), results[myname])


# Get results
results = dict()
datadir = join(DIR, "data")
for f in os.listdir(datadir):
    #np.savetxt('outfile.txt', np.array(r).flatten().view(float))
    glob = globals()
    if f.endswith(".txt") and f[:-4] in list(glob.keys()):
        results[f[:-4]] = np.loadtxt(join(datadir, f))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
