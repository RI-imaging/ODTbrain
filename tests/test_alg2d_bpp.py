"""Test 2d backpropagation"""
import sys

import numpy as np
import odtbrain

from common_methods import create_test_sino_2d, cutout, \
    get_test_parameter_set, write_results, get_results

WRITE_RES = False


def test_2d_backprop_phase():
    myframe = sys._getframe()
    sino, angles = create_test_sino_2d()
    parameters = get_test_parameter_set(2)
    r = list()
    for p in parameters:
        f = odtbrain.backpropagate_2d(sino, angles, **p)
        r.append(cutout(f))
    if WRITE_RES:
        write_results(myframe, r)
    assert np.allclose(np.array(r).flatten().view(float), get_results(myframe))


def test_2d_backprop_full():
    myframe = sys._getframe()
    sino, angles = create_test_sino_2d(ampl_range=(0.9, 1.1))
    parameters = get_test_parameter_set(2)
    r = list()
    for p in parameters:
        f = odtbrain.backpropagate_2d(sino, angles, **p)
        r.append(cutout(f))
    if WRITE_RES:
        write_results(myframe, r)
    assert np.allclose(np.array(r).flatten().view(float), get_results(myframe))


def test_2d_backprop_real():
    """
    Check if the real reconstruction matches the real part
    of the complex reconstruction.
    """
    sino, angles = create_test_sino_2d()
    parameters = get_test_parameter_set(2)
    # complex
    r = list()
    for p in parameters:
        f = odtbrain.backpropagate_2d(sino, angles, padval=0, **p)
        r.append(f)
    # real
    r2 = list()
    for p in parameters:
        f = odtbrain.backpropagate_2d(sino, angles, padval=0,
                                      onlyreal=True, **p)
        r2.append(f)
    assert np.allclose(np.array(r).real, np.array(r2))


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
