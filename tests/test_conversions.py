#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests refractive index conversion techniques
"""
from __future__ import division, print_function

import numpy as np
import os
from os.path import abspath, dirname, join, split
import sys

# Add parent directory to beginning of path variable
DIR = dirname(abspath(__file__))
sys.path = [split(DIR)[0]] + sys.path

import odtbrain._br

from common_methods import write_results, get_results

WRITE_RES = False


def get_test_data_set():
    """returns 3D array and parameters"""
    ln = 10
    f = np.arange(ln**3).reshape(ln,ln,ln)
    f = f + np.linspace(1,2,ln)
    res = 7
    nm = 1.34
    return f, res, nm


def get_test_data_set_sino(rytov=False):       
    """returns 3D array"""
    ln = 10
    a = 2
    sino = np.arange(ln*ln*a).reshape(a,ln,ln) /(ln*ln*a) * np.exp(1j*np.arange(ln*ln*a).reshape(a,ln,ln))
    if rytov:
        sino[0,0,0] = .1
    return sino


def negative_modulo_rest(a,b):
    """returns modulo with closest result to zero"""
    q = np.array(a / b, dtype=int)
    r = a - b * q

    # make sure r is close to zero
    wrong = np.where(np.abs(r)>b/2)
    r[wrong] -= b * np.sign(r[wrong])
    return r

def negative_modulo_rest_imag(x,b):
    """only modulo the imaginary part"""
    a = x.imag
    return x.real + 1j*negative_modulo_rest(a,b)
    
    

def test_odt_to_ri():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    f, res, nm = get_test_data_set()
    ri = odtbrain._br.odt_to_ri(f=f, res=res, nm=nm)
    if WRITE_RES:
        write_results(myframe, ri)
    assert np.allclose(np.array(ri).flatten().view(float), get_results(myframe))
    # Also test 2D version
    ri2d = odtbrain._br.odt_to_ri(f=f[0], res=res, nm=nm)
    assert np.allclose(ri2d, ri[0])


def test_opt_to_ri():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    f, res, nm = get_test_data_set()
    ri = odtbrain._br.opt_to_ri(f=f, res=res, nm=nm)
    if WRITE_RES:
        write_results(myframe, ri)
    assert np.allclose(np.array(ri).flatten().view(float), get_results(myframe))
    # Also test 2D version
    ri2d = odtbrain._br.opt_to_ri(f=f[0], res=res, nm=nm)
    assert np.allclose(ri2d, ri[0])


def test_sino_radon():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino = get_test_data_set_sino()
    rad = odtbrain._br.sinogram_as_radon(sino)
    if WRITE_RES:
        write_results(myframe, rad)
    assert np.allclose(np.array(rad).flatten().view(float), get_results(myframe))
    # Check the 3D result with the 2D result. They should be the same except
    # for a multiple of 2PI offset, because odtbrain._br._align_unwrapped
    # subtracts the background such that the minimum phase change is closest
    # to zero.
    # 2D A
    twopi = 2*np.pi
    rad2d = odtbrain._br.sinogram_as_radon(sino[:,:,0])
    assert np.allclose(0, negative_modulo_rest(rad2d - rad[:,:,0], twopi), atol=1e-6)
    # 2D B
    rad2d2 = odtbrain._br.sinogram_as_radon(sino[:,0,:])
    assert np.allclose(0, negative_modulo_rest(rad2d2 - rad[:,0,:], twopi), atol=1e-6)


def test_sino_rytov():
    myframe = sys._getframe()
    myname = myframe.f_code.co_name
    print("running ", myname)
    sino = get_test_data_set_sino(rytov=True)
    ryt = odtbrain._br.sinogram_as_rytov(sino)
    if WRITE_RES:
        write_results(myframe, ryt)
    assert np.allclose(np.array(ryt).flatten().view(float), get_results(myframe))
    # Check the 3D result with the 2D result. They should be the same except
    # for a multiple of 2PI offset, because odtbrain._br._align_unwrapped
    # subtracts the background such that the minimum phase change is closest
    # to zero.
    # 2D A
    twopi = 2*np.pi
    ryt2d = odtbrain._br.sinogram_as_rytov(sino[:,:,0])
    assert np.allclose(0, negative_modulo_rest_imag(ryt2d - ryt[:,:,0], twopi).view(float), atol=1e-6)
    # 2D B
    ryt2d2 = odtbrain._br.sinogram_as_rytov(sino[:,0,:])
    assert np.allclose(0, negative_modulo_rest_imag(ryt2d2 - ryt[:,0,:], twopi).view(float), atol=1e-6)



if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
    
