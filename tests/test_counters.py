"""Tests progress counters"""
import multiprocessing as mp

import numpy as np

import odtbrain

from common_methods import create_test_sino_2d, create_test_sino_3d, \
    get_test_parameter_set


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
    sino, angles = create_test_sino_3d(Nx=10, Ny=10)
    p = get_test_parameter_set(1)[0]
    # complex
    jmc = mp.Value("i", 0)
    jmm = mp.Value("i", 0)
    odtbrain.backpropagate_3d(sino, angles, padval=0,
                              dtype=np.float64,
                              count=jmc,
                              max_count=jmm,
                              **p)
    assert jmc.value == jmm.value
    assert jmc.value != 0


def test_back3d_tilted():
    sino, angles = create_test_sino_3d(Nx=10, Ny=10)
    p = get_test_parameter_set(1)[0]
    # complex
    jmc = mp.Value("i", 0)
    jmm = mp.Value("i", 0)
    odtbrain.backpropagate_3d_tilted(sino, angles, padval=0,
                                     dtype=np.float64,
                                     count=jmc,
                                     max_count=jmm,
                                     **p)
    assert jmc.value == jmm.value
    assert jmc.value != 0


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
