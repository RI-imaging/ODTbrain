#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests backpropagation algorithm
"""
from __future__ import division, print_function

import numpy as np
import os
from os.path import abspath, dirname, join, split
import sys

# Add parent directory to beginning of path variable
DIR = dirname(abspath(__file__))
sys.path = [split(DIR)[0]] + sys.path

import odtbrain
import odtbrain._Back_2D
import odtbrain._Back_3D
import odtbrain._br


def get_2d_test_sinogram(A=5, N=13):
    phase = np.exp(np.linspace(.1, 6, A*N))
    ampl = np.arange(.7, 1.5, A*N)
    sino = (ampl * np.exp(1j*phase)).reshape(A,N)
    angles = np.linspace(0,2*np.pi,A)
    return sino, angles


def get_3d_test_sinogram(A=5, M=11, N=13):
    phase = np.exp(np.linspace(.1, .3, A*M*N))
    ampl = np.arange(.7, 1.5, A*M*N)
    sino = (ampl * np.exp(1j*phase)).reshape(A,M,N)
    angles = np.linspace(0,2*np.pi,A)
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


def test_2d_backpropagate():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    sino, angles = get_2d_test_sinogram()
    parameters = get_test_parameter_set(2)
    r = list()
    for p in parameters:
        r.append(odtbrain._Back_2D.backpropagate_2d(sino, angles, **p))
    assert np.allclose(np.array(r).flatten().view(float), results[myname])


def test_3d_backpropagate():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    sino, angles = get_3d_test_sinogram()
    parameters = get_test_parameter_set(2)
    r = list()
    for p in parameters:
        r.append(odtbrain._Back_3D.backpropagate_3d(sino, angles, padval=0, **p))
    ri = odtbrain._br.odt_to_ri(np.array(r), 5.0, 1.4)
    assert np.allclose(np.array(ri).flatten().view(float), results[myname], atol=5e-6)


def test_3d_backpropagate32():
    # Check if float32 operations also go through
    myname = sys._getframe().f_code.co_name.strip("32")
    print("running ", myname)
    sino, angles = get_3d_test_sinogram()
    parameters = get_test_parameter_set(2)
    r = list()
    for p in parameters:
        r.append(odtbrain._Back_3D.backpropagate_3d(sino, angles,
            dtype=np.float32, padval=0, **p))
    ri = odtbrain._br.odt_to_ri(np.array(r), 5.0, 1.4)
    assert np.allclose(np.array(ri).flatten().view(np.float32),
                       results[myname],
                       atol=5e-6)


def test_3d_mprotate():
    myname = sys._getframe().f_code.co_name
    print("running ", myname)
    import ctypes
    import multiprocessing as mp

    ln = 10
    ln2 = 2*ln
    initial_array = np.arange(ln2**3).reshape((ln2,ln2,ln2))
    shared_array_base = mp.Array(ctypes.c_double, ln2 * ln2 * ln2)
    _shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    _shared_array = _shared_array.reshape(ln2, ln2, ln2)
    _shared_array[:,:,:] = initial_array
    odtbrain._shared_array = _shared_array
    # pool must be created after _shared array
    pool = mp.Pool(processes=mp.cpu_count())
    odtbrain._Back_3D._mprotate(2, ln, pool, 2)
    assert np.allclose(np.array(_shared_array).flatten().view(float), results[myname])
 

# Get results
results = dict()
datadir = join(DIR, "data")
for f in os.listdir(datadir):
    #np.savetxt('outfile.txt', np.array(r).flatten().view(float), fmt="%.10f")
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
    
