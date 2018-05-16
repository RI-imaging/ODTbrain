"""Test 3D backpropagation algorithm"""
import ctypes
import multiprocessing as mp
import platform
import sys

import numpy as np

import odtbrain
from odtbrain import alg3d_bpp

from common_methods import create_test_sino_3d, cutout, \
    get_test_parameter_set, write_results, get_results

WRITE_RES = False


def test_3d_backprop_phase():
    myframe = sys._getframe()
    sino, angles = create_test_sino_3d()
    parameters = get_test_parameter_set(2)
    r = list()
    for p in parameters:
        f = odtbrain.backpropagate_3d(sino, angles, padval=0,
                                      dtype=np.float64, **p)
        r.append(cutout(f))
    if WRITE_RES:
        write_results(myframe, r)
    data = np.array(r).flatten().view(float)
    assert np.allclose(data, get_results(myframe))
    return data


def test_3d_backprop_nopadreal():
    """
    - no padding
    - only real result
    """
    platform.system = lambda: "Windows"
    myframe = sys._getframe()
    sino, angles = create_test_sino_3d()
    parameters = get_test_parameter_set(2)
    r = list()
    for p in parameters:
        f = odtbrain.backpropagate_3d(sino, angles, padding=(False, False),
                                      dtype=np.float64, onlyreal=True, **p)
        r.append(cutout(f))
    if WRITE_RES:
        write_results(myframe, r)
    data = np.array(r).flatten().view(float)
    assert np.allclose(data, get_results(myframe))


def test_3d_backprop_windows():
    """
    We assume that we are not running these tests on windows.
    So we perform a test with fake windows to increase coverage.
    """
    datalin = test_3d_backprop_phase()
    real_system = platform.system
    datawin = test_3d_backprop_phase()
    platform.system = real_system
    assert np.allclose(datalin, datawin)


def test_3d_backprop_real():
    """
    Check if the real reconstruction matches the real part
    of the complex reconstruction.
    """
    sino, angles = create_test_sino_3d(Nx=10, Ny=10)
    parameters = get_test_parameter_set(2)
    # complex
    r = list()
    for p in parameters:
        f = odtbrain.backpropagate_3d(sino, angles, padval=0,
                                      dtype=np.float64,
                                      onlyreal=False, **p)
        r.append(f)
    # real
    r2 = list()
    for p in parameters:
        f = odtbrain.backpropagate_3d(sino, angles, padval=0,
                                      dtype=np.float64,
                                      onlyreal=True, **p)
        r2.append(f)
    assert np.allclose(np.array(r).real, np.array(r2))


def test_3d_backprop_phase32():
    sino, angles = create_test_sino_3d()
    parameters = get_test_parameter_set(2)
    r = list()
    for p in parameters:
        f = odtbrain.backpropagate_3d(sino, angles,
                                      dtype=np.float32, padval=0, **p)
        r.append(cutout(f))
    data32 = np.array(r).flatten().view(np.float32)
    data64 = test_3d_backprop_phase()
    assert np.allclose(data32, data64, atol=6e-7, rtol=0)


def test_3d_mprotate():
    myframe = sys._getframe()
    ln = 10
    ln2 = 2*ln
    initial_array = np.arange(ln2**3).reshape((ln2, ln2, ln2))
    shared_array_base = mp.Array(ctypes.c_double, ln2 * ln2 * ln2)
    _shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    _shared_array = _shared_array.reshape(ln2, ln2, ln2)
    _shared_array[:, :, :] = initial_array
    odtbrain._shared_array = _shared_array
    # pool must be created after _shared array
    pool = mp.Pool(processes=mp.cpu_count())
    alg3d_bpp._mprotate(2, ln, pool, 2)
    if WRITE_RES:
        write_results(myframe, _shared_array)
    assert np.allclose(np.array(_shared_array).flatten().view(
        float), get_results(myframe))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()