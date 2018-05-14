"""Tests progress counters"""

import multiprocessing as mp
import platform
import sys
import warnings
import zipfile

import numpy as np
from scipy.ndimage import rotate


import odtbrain
import odtbrain._Back_3D
import odtbrain._Back_3D_tilted
import odtbrain._br

from common_methods import create_test_sino_2d, create_test_sino_3d, create_test_sino_3d_tilted, cutout, get_test_parameter_set, write_results, get_results, normalize


def test_integrate_2d():
    sino, angles = create_test_sino_2d(N=10)
    p = get_test_parameter_set(1)[0]
    # complex
    jmc = mp.Value("i", 0)
    jmm = mp.Value("i", 0)

    odtbrain.integrate_2d(sino, angles,
                          count=jmc,
                          max_count=jmm,
                          **p)
    
    assert jmc.value == jmm.value
    assert jmc.value != 0


def test_fmp_2d():
    sino, angles = create_test_sino_2d(N=10)
    p = get_test_parameter_set(1)[0]
    # complex
    jmc = mp.Value("i", 0)
    jmm = mp.Value("i", 0)

    odtbrain.fourier_map_2d(sino, angles,
                            count=jmc,
                            max_count=jmm,
                            **p)
    
    assert jmc.value == jmm.value
    assert jmc.value != 0


def test_bpp_2d():
    sino, angles = create_test_sino_2d(N=10)
    p = get_test_parameter_set(1)[0]
    # complex
    jmc = mp.Value("i", 0)
    jmm = mp.Value("i", 0)

    odtbrain.backpropagate_2d(sino, angles, padval=0,
                              count=jmc,
                              max_count=jmm,
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